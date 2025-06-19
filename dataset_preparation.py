import torch
import torchvision.transforms as T
from einops import rearrange
from PIL import Image
import os

def load_and_preprocess(image_path, image_size=224, patch_size=16):
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    image = Image.open(image_path).convert('L')
    image = transform(image)
    patches = rearrange(image, 'c (h p1) (w p2) -> (h w) (p1 p2)', p1=patch_size, p2=patch_size)
    return patches  # N x P^2
