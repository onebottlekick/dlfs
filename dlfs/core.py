import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        
        
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward():
        raise NotImplementedError()
    

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.input.data
        gx = 2*x*gy
        return gx
    

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x)*gy
        return gx