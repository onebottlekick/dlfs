import unittest

import numpy as np

from dlfs import Variable

class VariableTest(unittest.TestCase):
    def test_dtype(self):
        data = np.array([[1, 2], [3, 4]])
        variable = Variable(data)

        self.assertTrue(data.dtype, variable.data.dtype)