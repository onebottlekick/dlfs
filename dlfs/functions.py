import numpy as np

from dlfs.core import Function, as_variable, exp, Config
from dlfs import utils
from dlfs import cuda
from utils import pair, get_conv_outsize


class Tanh(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
        return y
    
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy*(1 - y*y)
        return gx


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
        
    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y
    
    def backward(self, gy):
        return reshape(gy, self.x_shape)
    
    
class Transpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y
    
    def backward(self, gy):
        gx = transpose(gy)
        return gx
    

class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims    
    
    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        
        gx = broadcast_to(gy, self.x_shape)
        return gx


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape
        
    def forward(self, x):
        self.x_shape = x.shape
        xp = cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx
    

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape
        
    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx
    
    
class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y
    
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff**2).sum()/len(diff)
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy*diff*(2./len(diff))
        gx1 = -gx0
        return gx0, gx1

class ReLU(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.maximum(x, 0.0)
        return y
    
    def backward(self, gy):
        x,  = self.inputs
        mask = x.data > 0
        gx = gy*mask
        return gx
    

class Sigmoid(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x*0.5)*0.5 + 0.5
        return y
    
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy*y(1 - y)
        return gx
    

class Sin(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.sin(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy*cos(x)
        return gx
    

class Cos(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.cos(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy*(-sin(x))
        return gx
    

class Conv2d(Function):
    def __init__(self, stride=1, padding=0):
        super().__init__()
        
        self.stride = pair(stride)
        self.padding = pair(padding)
        
    def forward(self, x, W, b):
        xp = cuda.get_array_module(x)
        
        KH, KW = W.shape[2:]
        col = im2col_array(x, (KH, KW), self.stride, self.padding, to_matrix=False)
        
        y = xp.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            y += b
        y = xp.rollaxis(y, 3, 1)
        return y
    
    # TODO deconv2d, Conv2dGradW
    def backward(self, gy):
        x, W, b = self.inputs
        gx = deconv2d(gy, W, b=None, stride=self.stride, padding=self.padding)
        gW = Conv2dGradW(self)(x, gy)
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb


def tanh(x):
    return Tanh()(x)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


def transpose(x):
    return Transpose()(x)


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


def matmul(x, W):
    return MatMul()(x, W)


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


def linear(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t
    y = t + b
    t.data = None
    return y


def softmax(x, axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y/sum_y


def relu(x):
    return ReLU()(x)


def dropout(x, dropout_ratio=0.5):
    x = as_variable(x)
    
    if Config.train:
        xp = cuda.get_array_module(x)
        mask = xp.random.rand(*x.shape) > dropout_ratio
        scale = xp.array(1.0 - dropout_ratio).astype(x.dtype)
        y = x*mask/scale
        return y
    else:
        return x
    
    
def sin(x):
    return Sin()(x)


def cos(x):
    return Cos()(x)


def conv2d(x, W, b=None, stride=1, padding=0):
    return Conv2d(stride, padding)(x, W, b)


def im2col_array(img, kernel_size, stride, padding, to_matrix=True):
    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(padding)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)
    
    xp = cuda.get_array_module(img)
    
    # TODO _im2col_gpu
    if xp != np:
        col = _im2col_gpu(img, kernel_size, stride, padding)
    else:
        img = np.pad(img, ((0, 0), (0, 0), (PH, PH + SH -1), (PW, PW + SW -1)), mode='constant', constant_values=(0, ))
        col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)

        for j in range(KH):
            j_lim = j + SH*OH
            for i in range(KW):
                i_lim = i + SW*OW
                col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]
    
    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N*OH*OW, -1))
        
    return col