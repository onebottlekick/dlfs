import unittest

import numpy as np

from dlfs import Variable, Square, Exp


x = Variable(np.array(0.5))


def numerical_gradient(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)


class GradientTest(unittest.TestCase):
    
    def test_gradient(self):
        A = Square()
        B = Exp()
        C = Square()
        a = A(x)
        b = B(a)
        y = C(b)
        
        y.grad = 1.0
        b.grad = C.backward(y.grad)
        a.grad = B.backward(b.grad)
        x.grad = A.backward(a.grad)

        self.assertAlmostEqual(x.grad, numerical_gradient(lambda x: C(B(A(x))), x), places=6)
        
    def test_gradient2(self):
        A = Square()
        B = Exp()
        C = Square()
        a = A(x)
        b = B(a)
        y = C(b)
        
        y.grad = np.array(1.0)
        y.backward()

        self.assertAlmostEqual(x.grad, numerical_gradient(lambda x: C(B(A(x))), x), places=6)