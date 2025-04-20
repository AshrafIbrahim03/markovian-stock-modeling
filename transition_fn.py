import pandas as pd
import numpy as np
import get_state
import datetime
import time
from collections import Counter
import json


#  The below Combination class will be replaced by the State object we create.
#  As discussed, the state object must hold a pd.Series of signed integers.
#  Parameter: "bins" (I'm open to better names here) number of bins to create on the normalized scale. MUST BE ODD
class Combination:
    """
    This class takes in a series of normalized scores and assigns each score a value depending on the num_bins.
    State length is determined by the length of series, and the number of attributes is determined by num_bins - 1.
    TODO: make this work for num_bins >= 11 by overriding comparator and equals
    """

    def __init__(self,
                 series: pd.Series,
                 num_bins: int = 4):
        # key = {np.linspace(-100, 100, num=num_bins)[i]: i for i in range(num_bins)}
        thresholds = pd.Series(np.linspace(-100, 100, num=num_bins))
        self.combination = tuple(np.digitize(series, thresholds, right=False))

    def get_combination(self):
        return self.combination

    def __str__(self):
        return str(self.combination)


# For testing purposes, it is ok to use the below transition function with all default args
def transition_fn_by_prob(year_horizon: int = 2015,
                          path: str = './data/inflation_adjusted_berkshire_stocks.csv',
                          col: str = 'Open_adjusted',
                          days: int = 3,
                          state_func=get_state.get_state_by_percentile
                          ) -> []:
    """
    This function returns a transition probability matrix based on the frequency of past transitions.

    year_horizon: earliest year to include in assessment of transition history
    path: path to csv file containing numeric values
    col: string name of column within the csv located at path that contains the desired values
    days: number of days included in each state; aka "state length"
    """
    # print('pre-read: ' + str(time.time()))
    year_horizon = datetime.datetime(year_horizon, 1, 1)
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'] > year_horizon]

    # print('pre-data_series: ' + str(time.time()))
    data_list = [df[col][i:i + days] for i in range(len(df) - days)]
    data_series = pd.Series(data_list)

    # print('pre-state: ' + str(time.time()))
    state_history = [Combination(state_func
                                 (data_series=state, state_width=1, min=state.min(),
                                  max=state.max())).get_combination() for state in data_series]
    # print('post-state: ' + str(time.time()))
    combos = list({state for state in state_history})
    # print(combos)  # Only 12 of 27 possible combos are appearing
    # print('post-combos: ' + str(time.time()))
    #
    # state_history = pd.Series(state_history)
    # counts = state_history[:-1].value_counts()
    counts = Counter(state_history[:-1])  # How many times each state appears in the history
    # print('post state history series: ' + str(time.time()))

    # combos = state_history.unique()  # returns np array

    pairs = list(zip(state_history[:-1], state_history[1:]))
    # print(pairs)
    # print('post zip: ' + str(time.time()))

    data = [[pairs.count((i, j)) / counts[i] if counts[i] > 0 else 0
             for j in combos]
            for i in combos]
    result_df = pd.DataFrame(data, index=combos, columns=combos)
    return result_df
    # result_df.to_csv('transition_table.csv')


# print(transition_fn_by_prob())
