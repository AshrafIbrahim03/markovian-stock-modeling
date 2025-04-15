import pandas as pd
import numpy as np
import get_state
import datetime
import json


class Combination:
    """
    This class takes in a series of normalized scores and assigns each score a value depending on the num_bins.
    State length is determined by the length of series, and the number of attributes is determined by num_bins - 1.
    TODO: make this work for num_bins >= 11 by overriding comparator and equals
    """
    def __init__(self, series: pd.Series, num_bins: int = 4):
        # key = {np.linspace(-100, 100, num=num_bins)[i]: i for i in range(num_bins)}
        thresholds = pd.Series(np.linspace(-100, 100, num=num_bins))
        self.combination = np.digitize(series, thresholds, right=False)

    def __str__(self):
        return str(self.combination)


def transition_fn_by_prob(year_horizon: int = 2015,
                          path: str = './data/inflation_adjusted_berkshire_stocks.csv',
                          col: str = 'Open_adjusted',
                          days: int = 3
                          ) -> []:
    """
    This function returns a transition probability matrix based on the frequency of past transitions.

    year_horizon: earliest year to include in assessment of transition history
    path: path to csv file containing numeric values
    col: string name of column within the csv located at path that contains the desired values
    days: number of days included in each state; aka "state length"
    """
    year_horizon = datetime.datetime(year_horizon, 1, 1)
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'] > year_horizon]

    data_list = [df[col][i:i+days] for i in range(len(df)-days)]
    data_series = pd.Series(data_list)

    state_history = [str(Combination(get_state.get_state_by_percentile
                                     (data_series=state, state_width=1, min=state.min(),
                                      max=state.max()))) for state in data_series]
    state_history = pd.Series(state_history)
    counts = state_history[:-1].value_counts()

    combos = sorted(state_history.unique())
    combo_series = pd.Series(combos)

    hist_list = state_history.to_list()
    pairs = list(zip(hist_list[:-1], hist_list[1:]))

    data = [[pairs.count((i, j)) / counts[i] if counts[i] > 0 else 0
             for j in combos]
            for i in combos]
    result_df = pd.DataFrame(data, index=combo_series, columns=combo_series)
    result_df.to_csv('transition_table.csv')


transition_fn_by_prob()