from PIL import Image

import numpy as np


class Compose:
    def __init__(self, transforms=[]):
        self.transforms = transforms
        
    def __call__(self, img):
        if not self.transforms:
            return img
        for t in self.transforms:
            img = t(img)
        return img
    
    
class ToArray:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype
        
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return img
        if isinstance(img, Image.Image):
            img = np.asarray(img)
            img = img.transpose(2, 0, 1)
            img = img.astype(self.dtype)
            return img
        else:
            raise TypeError
        
        
class Resize:
    def __init__(self, size, mode=Image.BILINEAR):
        def pair(x):
            if isinstance(x, int):
                return (x, x)
            elif isinstance(x, tuple):
                assert len(x) == 2
                return x
            else:
                return ValueError
        self.size = pair(size)
        self.mode = mode
        
    def __call__(self, img):
        return img.resize(self.size, self.mode)