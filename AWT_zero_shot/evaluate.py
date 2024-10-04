import argparse
import time
import os
import json

import torch
import torch.nn.functional as F

from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, set_random_seed
from data.cls_to_names import get_classnames, CUSTOM_TEMPLATES

from clip import clip
def load_clip_to_cpu(arch):
    url = clip._MODELS[arch]
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model

def calculate_batch_entropy(logits):
    return -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)

@torch.no_grad()
def get_entropy_weight(output, img_t=0.5, text_t=0.5):
    with torch.cuda.amp.autocast():
        # get weights for images
        image_entropy = calculate_batch_entropy(output.mean(1))
        image_weights = F.softmax(-image_entropy/img_t, dim=-1)

        # get weights for descriptors
        _, n_des, n_cls = output.shape
        anchor = output[0].mean(0)[None, None, :].repeat(n_des, n_cls, 1)
        output_des = output[0].unsqueeze(-1)
        scatter_indices = torch.arange(n_cls)[None, :, None].repeat(n_des, 1, 1).cuda()
        anchor.scatter_(dim=2, index=scatter_indices, src=output_des) # n_des, n_cls, n_cls
        text_entropy = calculate_batch_entropy(anchor)
        text_weights = F.softmax(-text_entropy.t()/text_t, dim=-1) # n_cls, n_des

    return image_weights, text_weights

def Sinkhorn(K, u, v):
    r = torch.ones_like(u)
    c = torch.ones_like(v)
    thresh = 1e-2
    for i in range(100):
        r0 = r
        r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break
    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
    return T

def optimal_transport(logits, logit_scale, image_weights, text_weights):
    eps = 0.1
    sim = logits / logit_scale.exp()
    sim = sim.permute(2, 0, 1) # n_cls x M x N

    wdist = 1.0 - sim
    with torch.no_grad():
        KK = torch.exp(-wdist / eps)
        T = Sinkhorn(KK, image_weights, text_weights)
        T = T.permute(1, 2, 0)
    assert not torch.isnan(T).any()

    return torch.sum(T * logits, dim=(0, 1)).unsqueeze(0)


@torch.no_grad()
def AWT_evaluation(clip_model, args):
    dataset_name = args.test_set
    print("Evaluating: {}".format(dataset_name))
    classnames = get_classnames(dataset_name)

    # get LLM descriptors
    if dataset_name in ['imagenet', 'imagenet_a', 'imagenetv2']:
        description_file = os.path.join(args.descriptor_path, 'imagenet.json')
    else:
        description_file = os.path.join(args.descriptor_path, f'{dataset_name}.json')
    print(f'Using description file: {description_file}')
    llm_descriptions = json.load(open(description_file))

    # ============== prepare text features ==============
    text_features = []
    template = CUSTOM_TEMPLATES[dataset_name]
    for classname in classnames:
        prompts = []
        prompt = template.format(classname.replace("_", " "))
        prompts.append(prompt + '.')

        # get descriptions
        assert len(llm_descriptions[classname]) >= args.num_descriptor
        for i in range(args.num_descriptor):
            prompt_desc = prompt + '. ' + llm_descriptions[classname][i]
            prompts.append(prompt_desc)
        prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()

        with torch.cuda.amp.autocast():
            text_features.append(clip_model.encode_text(prompts)) # n_desc x d

    text_features = torch.cat(text_features).float() # (n_cls x n_desc) x d
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # ==============  calculate logits for each image ==============
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)

    pre_features = torch.load(f"./pre_extracted_feat/{args.arch.replace('/', '')}/seed{args.seed}/{dataset_name}.pth")

    print("number of test samples: {}".format(len(pre_features)))
    progress = ProgressMeter(len(pre_features), [batch_time, top1], prefix='Test: ')

    end = time.time()
    for i, (image_features, target) in enumerate(pre_features):
        n_views = image_features.size(0)
        n_prompt = args.num_descriptor + 1

        output = clip_model.logit_scale.exp() * image_features @ text_features.t()
        output = output.view(n_views, -1, n_prompt).permute(0, 2, 1).contiguous() # n_view x n_prompt x c

        image_temperature = 0.5
        text_temperature = 0.5
        image_weights, text_weights = get_entropy_weight(output, img_t=image_temperature, text_t=text_temperature)
        output_ot = optimal_transport(output, clip_model.logit_scale, image_weights, text_weights)

        # measure accuracy
        acc1, = accuracy(output_ot, target, topk=(1,))
        top1.update(acc1[0], 1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i)

    print(f'\n *  {dataset_name}')
    progress.display_summary()


def main_worker(args):
    print("=> Model created: visual backbone {}".format(args.arch))
    clip_model = load_clip_to_cpu(args.arch)
    clip_model = clip_model.cuda()
    clip_model.float()
    clip_model.eval()

    for _, param in clip_model.named_parameters():
        param.requires_grad_(False)
    
    # testing start
    AWT_evaluation(clip_model, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AWT evaluation')
    parser.add_argument('--test_set', type=str, help='dataset name')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16')
    parser.add_argument('-p', '--print-freq', default=1000, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--descriptor_path', type=str)
    parser.add_argument('--num_descriptor', type=int, default=50)

    args = parser.parse_args()
    set_random_seed(args.seed)
    main_worker(args)