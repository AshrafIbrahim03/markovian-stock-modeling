import unittest
import json
import numpy as np
import pandas as pd
import transition_fn
import os


class MyTestCase(unittest.TestCase):
    def test_row_sum(self):
        current_dir = os.path.dirname(__file__)
        root = os.path.abspath(os.path.join(current_dir, '..'))
        data_path = os.path.join(root, 'data', 'inflation_adjusted_berkshire_stocks.csv')
        result = transition_fn.transition_fn_by_prob(path=data_path)
        check = result.sum(axis=1)
        assert np.isclose(check, 1, atol=1e-6).all()

    def test_illegal_states(self):
        current_dir = os.path.dirname(__file__)
        root = os.path.abspath(os.path.join(current_dir, '..'))
        data_path = os.path.join(root, 'data', 'inflation_adjusted_berkshire_stocks.csv')

        result = transition_fn.transition_fn_by_prob(path=data_path)
        print(result)

if __name__ == '__main__':
    unittest.main()
