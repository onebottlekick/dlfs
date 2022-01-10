import unittest

import numpy as np

from dlfs import Variable, Square, Exp

class VariableTest(unittest.TestCase):
    def test_dtype(self):
        data = np.array([[1, 2], [3, 4]])
        variable = Variable(data)

        self.assertTrue(data.dtype, variable.data.dtype)
        
    def test_variable_creator(self):
        A = Square()
        B = Exp()
        C = Square()
        
        x = Variable(np.array(0.5))
        a = A(x)
        b = B(a)
        y = C(b)
        
        self.assertTrue(y.creator, C)
        self.assertTrue(y.creator.input, b)
        self.assertTrue(b.creator, B)