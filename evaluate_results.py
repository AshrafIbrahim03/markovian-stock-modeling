import pandas as pd
import datetime
import chains
from chains import MarkovChain
import transition_fn
import get_state
import get_target
import os

df = pd.read_csv('/Users/walkerwatson/PycharmProjects/markovian-stock-modeling/data/inflation_adjusted_berkshire_stocks.csv')
df = pd.Series(df['Open_adjusted'])
linreg = chains.LinRegMC(df, 3, num_bins=3)

class ChainEvaluator:
    """
    When supplied with a markov chain and a range of training days, this class represents a functioning prediction model.
    day_correct: returns a binary value representing a correct guess by the model
    set_params: Used to alter the parameters of the chain without discarding any progress recorded

    """
    results = pd.DataFrame(columns=['model', 'num_atts', 'window_length', 'day', 'correct'])
    num_atts = None
    window_length = None

    def __init__(self, chain, days):
        self.chain = chain
        self.days = days

    def evaluate(self):
        if isinstance(self.chain, MarkovChain):
            pass

    def set_params(self, num_atts, window_length):
        self.num_atts = num_atts
        self.window_length = window_length