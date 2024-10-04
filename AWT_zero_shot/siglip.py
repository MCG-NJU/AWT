import argparse
from PIL import Image
from tqdm import tqdm
import os
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.datautils import Augmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, set_random_seed
from data.cls_to_names import *
from evaluate import get_entropy_weight, Sinkhorn

from transformers import AutoTokenizer, AutoModel, AutoImageProcessor

def optimal_transport(logits, image_weights, text_weights):
    eps = 0.1
    sim = logits
    sim = sim.permute(2, 0, 1) # n_cls x M x N

    wdist = 1.0 - sim
    with torch.no_grad():
        KK = torch.exp(-wdist / eps)
        T = Sinkhorn(KK, image_weights, text_weights)
        T = T.permute(1, 2, 0)
    assert not torch.isnan(T).any()

    return torch.sum(T * logits, dim=(0, 1)).unsqueeze(0)


MODEL_PATH = 'google/siglip-base-patch16-224'
# MODEL_PATH = '/mnt/petrelfs/zhaozhiyu/zhuyuhan/.clip_ckpt/siglip-base-patch16-224'


@torch.no_grad()
def evaluate(val_loader, model, text_features, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    print("number of test samples: {}".format(len(val_loader)))
    progress = ProgressMeter(len(val_loader), [batch_time, top1], prefix='Test: ')

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        assert isinstance(images, list)
        # image_processor = AutoImageProcessor.from_pretrained("/mnt/petrelfs/zhaozhiyu/zhuyuhan/.clip_ckpt/siglip-base-patch16-224")
        image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
        inputs = image_processor(images=torch.cat(images), return_tensors="pt", do_rescale=False)
        inputs['pixel_values'] = inputs['pixel_values'].cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # optimal transport
        n_views = image_features.size(0)
        n_prompt = args.num_descriptor + 1

        output = image_features @ text_features.t()
        output = output.view(n_views, -1, n_prompt).permute(0, 2, 1).contiguous() # n_view x n_prompt x c

        image_weights, text_weights = get_entropy_weight(output * model.logit_scale.exp() + model.logit_bias)
        output_ot = optimal_transport(output, image_weights, text_weights)

        acc1, = accuracy(output_ot, target, topk=(1,))
        top1.update(acc1[0], 1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i)

    print(f'\n *  {args.test_set}')
    progress.display_summary()
    with open(f'./results/siglip/all.txt', 'a') as f:
        acc_str = " ".join([meter.summary() for meter in progress.meters])
        acc = float(acc_str[7:])
        f.write(f'{acc}\t{args.test_set}\n')


@torch.no_grad()
def main_worker(args):
    # Image Transformation
    base_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=BICUBIC),
        transforms.CenterCrop(args.resolution)])
    preprocess = transforms.Compose([transforms.ToTensor()])
    # data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size, augmix=False)
    data_transform = Augmenter(base_transform, preprocess, n_views=args.batch_size)

    # Build dataloader
    val_dataset = build_dataset(args.test_set, data_transform, args.data, mode='test')
    print("number of test samples: {}".format(len(val_dataset)))
    val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=1, shuffle=True,
                num_workers=args.workers, pin_memory=True)

    print("Evaluating: {}".format(args.test_set))
    classnames = get_classnames(args.test_set)

    # get LLM descriptors
    dataset_name = args.test_set
    if dataset_name in ['imagenet', 'imagenet_a', 'imagenetv2']:
        description_file = os.path.join(args.descriptor_path, 'imagenet.json')
    else:
        description_file = os.path.join(args.descriptor_path, f'{dataset_name}.json')
    print(f'Using description file: {description_file}')
    llm_descriptions = json.load(open(description_file))

    # get models
    # model = AutoModel.from_pretrained("/mnt/petrelfs/zhaozhiyu/zhuyuhan/.clip_ckpt/siglip-base-patch16-224")
    model = AutoModel.from_pretrained(MODEL_PATH)
    model = model.cuda()
    # tokenizer = AutoTokenizer.from_pretrained("/mnt/petrelfs/zhaozhiyu/zhuyuhan/.clip_ckpt/siglip-base-patch16-224")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # prepare text features
    text_features = []
    template = CUSTOM_TEMPLATES[dataset_name]

    print('Preparing text features...')
    for classname in tqdm(classnames):
        prompts = []
        prompt = template.format(classname.replace("_", " "))
        prompts.append(prompt + '.')

        # get descriptions
        assert len(llm_descriptions[classname]) >= args.num_descriptor
        for i in range(args.num_descriptor):
            prompt_desc = prompt + '. ' + llm_descriptions[classname][i]
            prompts.append(prompt_desc)
        
        text_inputs = tokenizer(prompts, padding="max_length", return_tensors="pt", truncation=True)
        text_inputs['input_ids'] = text_inputs['input_ids'].cuda()

        with torch.cuda.amp.autocast():
            text_outputs = model.get_text_features(**text_inputs)
            text_features.append(text_outputs) # n_desc x d

    text_features = torch.cat(text_features).float() # (n_cls x n_desc) x d
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # start evaluate
    evaluate(val_loader, model, text_features, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AWT for SigLIP')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_set', type=str, help='dataset name')
    parser.add_argument('--resolution', default=224, type=int, help='image resolution')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=50, type=int, metavar='N')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-p', '--print-freq', default=1000, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--descriptor_path', type=str)
    parser.add_argument('--num_descriptor', type=int, default=50)

    args = parser.parse_args()
    set_random_seed(args.seed)
    main_worker(args)