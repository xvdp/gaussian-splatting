"""@ xvdp
Dataset to serve view points while loading cameras from disk
using torch Dataloader
"""
from typing import Union, Optional
import os
import os.path as osp
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset


# pylint: disable=no-member

class CamInfoDataset(Dataset):
    """ load viewpoints
    """
    def __init__(self, camera_list, root_dir):
        self.camera_list = camera_list
        self.names = {osp.splitext(f.name)[0]:f.path for f in os.scandir(root_dir)}

    def __len__(self):
        return len(self.camera_list)

    def __getitem__(self, idx):
        """return only what is needed for training"""
        out = self.camera_list[idx]
        image = image_8bit_to_tensor(self.names[out.image_name])
        return [image, out.world_view_transform, out.full_proj_transform,
                out.camera_center, out.FoVx, out.FoVy, out.image_width, out.image_height]


def image_8bit_to_tensor(image: Union[str, ImageFile.ImageFile, np.ndarray],
                         device: Union[torch.device, str, None] = None,
                         dtype: Optional[torch.dtype] = None ) -> torch.Tensor:
    """ converts, 8bit image path, PIL.Image, or ndarray to
    applies alpha if present
    out Tensor (1|3, H, W)
    Args:
        image   (str, PIL.Image, ndarray) 8 bit image
        devie   (str, torch.device [None]
    """
    if isinstance(image, str):
        image = Image.open(image)
    dtype = torch.get_default_dtype() if dtype is None else dtype
    image = torch.as_tensor(np.array(image), dtype=dtype,device=device) / 255.0
    if image.ndim == 3:
        image = image.permute(2,0,1).contiguous()
    elif image.ndim == 2:
        image = image[None]
    if len(image) == 4:
        image[:3] *= image[3:]
    return image[:3]
