import contextlib
import weakref

import numpy as np


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)
        

def no_grad():
    return using_config('enable_backprop', False)

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
        
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
        
    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()
        
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        
        add_func(self.creator)
        
        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
            
                if x.creator is not None:
                    add_func(x.creator)
                    
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None
                    
    def cleargrad(self):
        self.grad = None
        
        
class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        
        if Config.enable_backprop:        
            self.generation = max([x.generation for x in inputs])            
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self):
        raise NotImplementedError()
    

class Config:
    enable_backprop = True

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2*x*gy
        return gx
    

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x)*gy
        return gx
    

class Add(Function):
    def forward(self, *xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)

    def backward(self, gy):
        return gy, gy
    
    
def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


def add(xs):
    return Add()(xs)