import unittest

import numpy as np

from dlfs import Variable, Square, Exp, square, exp


x = Variable(np.array([10]))

class FunctionTest(unittest.TestCase):
    def test_square(self):
        y = Square()(x)
        self.assertTrue(x.data**2, y.data)        
        
    def test_exp(self):
        y = Exp()(x)
        self.assertTrue(np.exp(x.data), y.data)
        
    def test_composite_function(self):
        A = Square()
        B = Exp()
        C = Square()
        a = A(x)
        b = B(a)
        y = C(b)
        self.assertTrue(np.exp(x.data**2)**2, y.data)
        
    def test_short_function(self):
        a = square(x)
        self.assertTrue(x.data, 100)