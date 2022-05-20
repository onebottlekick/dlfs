import numpy as np

from dlfs.core import Function, as_variable, exp, Config
from dlfs import utils
from dlfs import cuda
from dlfs.utils import pair, get_conv_outsize, get_deconv_outsize


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
    
    def backward(self, gy):
        x, W, b = self.inputs
        gx = deconv2d(gy, W, b=None, stride=self.stride, padding=self.padding)
        gW = Conv2dGradW(self)(x, gy)
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb
    
    
class Deconv2d(Function):
    def __init__(self, stride=1, padding=0, outsize=None):
        super().__init__()
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.outsize = outsize
        
    def forward(self, x, W, b):
        xp = cuda.get_array_module(x)
        
        SH, SW = self.stride
        PH, PW = self.padding
        C, OC, KH, KW = W.shape
        N, C, H, W  =x.shape
        if self.outsize is None:
            OH = get_deconv_outsize(H, KH, SH, PH)
            OW = get_deconv_outsize(W, KW, SW, PW)
        else:
            OH, OW = pair(self.outsize)
        img_shape = (N, OC, OH, OW)
        
        gcol = xp.tensordot(W, x, (0, 1))
        gcol = xp.rollaxis(gcol, 3)
        y = col2im_array(gcol, img_shape, (KH, KW), self.stride, self.padding, to_matrix=False)
        
        if b is not None:
            self.no_bias = True
            y += b.reshape((1, b.size, 1, 1))
        
        return y
    
    def backward(self, gy):
        x, W, b = self.inputs
        gx = conv2d(gy, W, b=None, stride=self.stride, padding=self.padding)
        f = Conv2dGradW(self)
        gW = f(gy, x)
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb


class Conv2dGradW(Function):
    def __init__(self, conv2d):
        W = conv2d.inputs[1]
        kh, kw = W.shape[2:]
        self.kernel_size = (kh, kw)
        self.strde = conv2d.stride
        self.pad = conv2d.padding
        
    def forward(self, x, gy):
        xp = cuda.get_array_module(x)
        
        col = im2col_array(x, self.kernel_size, self.stride, self.padding, to_matrix=False)
        gW = xp.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))
        return gW
    
    def backward(self, gys):
        x, gy = self.inputs
        gW, = self.outputs
        xh, xw = x.shape[2:]
        gx = deconv2d(gy, gW, stride=self.stride, padding=self.padding, outsize=(xh, xw))
        ggy = conv2d(x, gW, stride=self.stride, padding=self.padding)
        return gx, ggy
    

class Pooling(Function):
    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.padding, to_maxrix=False)
        N, C, KH, KW, OH, OW = col.shape
        self.indexes = col.argmax(axis=2)
        y = col.max(axis=2)
        
        return y
    
    def backward(self, gy):
        return Pooling2DGrad(self)(gy)
    
    
# TODO Pooling2DWithIndexes
class Pooling2DGrad(Function):
    def __init__(self, mpool2d):
        self.mpool2d = mpool2d
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.padding = mpool2d.padding
        self.input_shape = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes
        
    def forward(self, gy):
        xp = cuda.get_array_module(gy)
        
        N, C, OH, OW = gy.shape
        N, C, H, W = self.input_shape
        KH, KW = pair(self.kernel_size)
        
        gcol = xp.zeros((N*C*OH*OW*KH*KW), dtype=self.dtype)
        
        indexes = (self.indexes.ravel() + xp.arange(0, self.indexes.size*KH*KW, KH*KW))

        gcol[indexes] = gy.ravel()
        gcol = gcol.reshape(N, C, OH, OW, KH, KW)
        gcol = xp.swapaxis(gcol, 2, 4)
        gcol = xp.swapaxis(gcol, 3, 5)
        
        gx = col2im_array(gcol, (N, C, H, W), self.kernel_size, self.stride, self.padding, to_matrix=False)
        
        return gx
    
    def backward(self, ggx):
        f = Pooling2DWithIndexes(self.mpool2d)
        return f(ggx)


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


def _im2col_gpu(img, kernel_size, stride, padding):
    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SY, SX = pair(stride)
    PH, PW = pair(padding)
    OH = get_conv_outsize(H, KH, SY, PH)
    OW = get_conv_outsize(W, KW, SX, PW)
    dy, dx = 1, 1
    col = cuda.cupy.empty((N, C, KH, KW, OH, OW), dtype=img.dtype)
    
    cuda.cupy.ElementwiseKernel(
        'raw T img, int32 H, int32 W, int32 OH, int32 OW,'
        'int32 KH, int32 KW, int32 SY, int32 SX, int32 PH, int32 PW,'
        'int32 dy, int32 dx',
        'T col',
        '''
           int c0 = i / (KH * KW * OH * OW);
           int ky = i / (KW * OH * OW) % KH;
           int kx = i / (OH * OW) % KW;
           int out_y = i / OW % OH;
           int out_x = i % OW;
           int in_y = ky * dy + out_y * SY - PH;
           int in_x = kx * dx + out_x * SX - PW;
           if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
             col = img[in_x + W * (in_y + H * c0)];
           } else {
             col = 0;
           }
        ''',
        'im2col'
        )(img.reduced_view(), H, W, OH, OW, KH, KW, SY, SX, PH, PW, dy, dx, col)
    
    return col


def deconv2d(x, W, b=None, stride=1, padding=0, outsize=None):
    return Deconv2d(stride, padding, outsize)(x, W, b)


def col2im_array(col, img_shape, kernel_size, stride, padding, to_matrix=True):
    N, C, H, W = img_shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(padding)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    xp = cuda.get_array_module(col)
    if xp != np:
        img = _col2im_gpu(col, SH, SW, PH, PW, H, W)
        return img
    
    else:
        img = np.zeros((N, C, H + 2*PH + SH, W + 2*PW + SW -1), dtype=col.dtype)
        
        for j in range(KH):
            j_lim = j + SH*OH
            for i in range(KW):
                i_lim = i + SW*OW
                img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]
        return img[:, :, PH:H+PH, PW:W+PW]
    
    
def _col2im_gpu(col, SY, SX, PH, PW, H, W):
    N, C, KH, KW, OH, OW = col.shape
    dx, dy = 1, 1
    img = cuda.cupy.empty((N, C, H, W), dtype=col.dtype)

    cuda.cupy.ElementwiseKernel(
            'raw T col, int32 H, int32 W, int32 OH, int32 OW,'
        'int32 KH, int32 KW, int32 SY, int32 SX, int32 PH, int32 PW,'
        'int32 dx, int32 dy',
        'T img',
        '''
           int c0 = i / (h * w);
           int y  = i / W % H;
           int x  = i % W;
           T val = 0;
           for (int ky = 0; ky < KH; ++ky) {
             int out_y = (y + PH - ky * dy);
             if (0 > out_y || out_y >= out_h * SY) continue;
             if (out_y % SY != 0) continue;
             out_y /= SY;
             for (int kx = 0; kx < kw; ++kx) {
               int out_x = (x + PW - kx * dx);
               if (0 > out_x || out_x >= OW * SX) continue;
               if (out_x % SX != 0) continue;
               out_x /= SX;
               int k = out_y + OH * (kx + kw * (ky + kh * c0));
               val = val + col[out_x + out_w * k];
             }
           }
           img = val;
        ''',
        'col2im')(col.reduced_view(), H, W, OH, OW, KH, KW, SY, SX, PH, PW, dx, dy, img)
    return img