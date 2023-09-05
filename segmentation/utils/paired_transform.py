import random
import torchvision.transforms.functional as ttf
import numpy as np

def unpaired_random_crop(img_lqs, img_refs, patch_size=(512,512)):

    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]
    if not isinstance(img_refs, list):
        img_refs = [img_refs]

    _, h_lq, w_lq = img_lqs[0].shape
    _, h_ref, w_ref = img_refs[0].shape
    assert h_lq == h_ref and w_lq == w_ref, 'The size of images should be equal.'

    if h_lq < patch_size[0] or w_lq < patch_size[1]:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({patch_size[0]}, {patch_size[1]}). ')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - patch_size[0])
    left = random.randint(0, w_lq - patch_size[1])
    img_lqs = [v[:,top:top + patch_size[0], left:left + patch_size[1]] for v in img_lqs]
    top = random.randint(0, h_ref - patch_size[0])
    left = random.randint(0, w_ref - patch_size[1])
    img_refs = [v[:,top:top + patch_size[0], left:left + patch_size[1]] for v in img_refs]

    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    if len(img_refs) == 1:
        img_refs = img_refs[0]
    return img_lqs, img_refs

def augment(imgs, hflip=True, rotation=True):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees) OR brightness OR saturation.

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[tensor] | tensor): Images to be augmented. If the input
            is an tensor, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[tensor]: Flows to be augmented. If the input is an
            tensor, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[tensor] | tensor: Augmented images and flows. If returned
            results only have one element, just return tensor.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            img = ttf.hflip(img)
        if vflip:  # vertical
            img = ttf.vflip(img)
        if rot90:
            img = img.permute(0, 2, 1)
        return img

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]

    return imgs