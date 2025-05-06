from unittest import TestCase

import pandas as pd

from chains import FreqModelMC
import get_state
import datetime


class TestFreqModelMC(TestCase):
    def test__step(self):
        df = pd.read_csv('/Users/walkerwatson/PycharmProjects/markovian-stock-modeling/data/inflation_adjusted_berkshire_stocks.csv')
        train_range = (datetime.datetime(2015, 1, 1), datetime.datetime(2021, 1, 1))
        sample_range = (datetime.datetime(2020, 1, 2), datetime.datetime(2021, 1, 1))
        chain = FreqModelMC(df=df, train_range=train_range, days=3, att_num=13, sample_range=sample_range, state_func=get_state.get_state_by_perc_change)
        print(chain.step(10))

