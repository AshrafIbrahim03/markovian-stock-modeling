import unittest
import json


class MyTestCase(unittest.TestCase):
    def test_something(self):
        with open("/Users/walkerwatson/PycharmProjects/markovian-stock-modeling1/transition_table.json", "r") as f:
            data = json.load(f)
        for k, v in data.items():
            total = 0
            for val in data[k].values():
                total += val
            print(total)
            self.assertAlmostEqual(total, 1, delta=.02)  # add assertion here


if __name__ == '__main__':
    unittest.main()
