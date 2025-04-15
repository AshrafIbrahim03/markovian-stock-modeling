import unittest
import json
import numpy as np
import pandas as pd


class MyTestCase(unittest.TestCase):
    def test_row_sum(self):
        data = pd.read_csv('/Users/walkerwatson/PycharmProjects/markovian-stock-modeling1/transition_table.csv',
                           index_col=0, header=0)
        check = data.sum(axis=1)
        assert np.isclose(check, 1, atol=1e-6).all()  # add assertion here


if __name__ == '__main__':
    unittest.main()
