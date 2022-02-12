import numpy as np

from core import Function


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y
    
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy*(1 - y*y)
        return gx
    

def tanh(x):
    return Tanh()(x)