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