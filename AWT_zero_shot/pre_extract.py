import argparse
from PIL import Image
from tqdm import tqdm
import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.datautils import Augmenter, build_dataset
from utils.tools import set_random_seed
from data.cls_to_names import *

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

@torch.no_grad()
def pre_extract_image_feature(val_loader, clip_model, args):

    save_dir = f"./pre_extracted_feat/{args.arch.replace('/', '')}/seed{args.seed}"
    os.makedirs(save_dir, exist_ok=True)

    all_data = []
    for images, target in tqdm(val_loader):
        assert isinstance(images, list)
        for k in range(len(images)):
            images[k] = images[k].cuda(non_blocking=True)
        images = torch.cat(images, dim=0)
        target = target.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        all_data.append((image_features, target))

    save_path = os.path.join(save_dir, f"{args.test_set}.pth")
    torch.save(all_data, save_path)
    print(f"Successfully save image features to [{save_path}]")


def main_worker(args):
    print("=> Model created: visual backbone {}".format(args.arch))
    clip_model = load_clip_to_cpu(args.arch)
    clip_model = clip_model.cuda()
    clip_model.float()
    clip_model.eval()

    for _, param in clip_model.named_parameters():
        param.requires_grad_(False)

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    
    base_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=BICUBIC),
        transforms.CenterCrop(args.resolution)])
    preprocess = transforms.Compose([transforms.ToTensor(), normalize])
    data_transform = Augmenter(base_transform, preprocess, n_views=args.batch_size)

    print("Extracting features for: {}".format(args.test_set))

    val_dataset = build_dataset(args.test_set, data_transform, args.data, mode='test')
    print("number of test samples: {}".format(len(val_dataset)))
    val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=1, shuffle=True,
                num_workers=args.workers, pin_memory=True)
    
    pre_extract_image_feature(val_loader, clip_model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-extracting image features')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_set', type=str, help='dataset name')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=50, type=int, metavar='N')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    set_random_seed(args.seed)
    main_worker(args)