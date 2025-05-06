import pandas as pd
import numpy as np

from state import State


def get_state_by_zscore(data_series:pd.Series,states:int = 6, state_width:float= 1,mean:float=0,stdev:float=20)-> pd.Series:
    """
        This function bins each value from `data_series` into `states` number of states where each state has a width of `state_width`. This is all done in terms of z score

        # Args:
        data_series(pd.Series): The values to bin. Will be clamped between [-(states/2 * state_width),(states/2 * state_width)]

        states(int): The number of states to bin into

        state_width(float): The width of each state, in z score, i.e. a state_width of one would mean that one state would be one z-score wide

        mean(float): The mean to compute z-scores with

        stdev(float): The standard deviation to compute z-scores with
    """
    z_scores = (data_series - mean) /stdev 
    scores = np.clip(z_scores,-(np.floor(states/2) * state_width),(np.floor(states/2) * state_width))
    bins = [-(states/2 * state_width) + (state_width * i) for i in range(states)]

    return  np.round(scores / state_width) * state_width



def get_state_by_percentile(data_series:pd.Series, states:int=6, state_width:float=1,min:float=0,max:float=100):
    """
            This function bins each value from `data_series` into `states` number of states where each state has a width of `state_width`. This is all done in terms of normalizing from `min` to `max`

            # Args:
            data_series(pd.Series): The values to bin. Will be clamped between [-(states/2 * state_width),(states/2 * state_width)]

            states(int): The number of states to bin into

            state_width(float): The width of each state, in z score, i.e. a state_width of one would mean that one state would be one as normalized from `max` to `min`

            min(float): The min to normalize with

            max(float): The max to normalize with
        """
    clamped = np.clip(data_series,min,max)
    percentiles = (clamped - min) / (max - min) * 100
    scores = np.clip(percentiles - 50, -50, 50)

    return np.round(scores / state_width) * state_width


def get_state_by_perc_change(data_series: pd.Series):
    """
        accepts a Series of historical value data (length n), returns a series of percent daily changes (length n-1)
        num_bins: optional arg to convert percent changes into state-ready int values. MUST BE ODD
        """
    start_days = data_series[:-1]
    end_days = data_series[1:]
    start_days.reset_index(drop=True)
    pairs = list(zip(start_days, end_days))
    diffs = [p[1] - p[0] for p in pairs]
    perc_change = (diffs[i] / start_days.iloc[i] for i in range(len(start_days)))
    perc_change = pd.Series(perc_change)
    return perc_change


def integerize_state(data_series: pd.Series, num_bins: int, max_delta: float) -> tuple:
    # TODO: investigate why output is
    if num_bins % 2 != 1:
        print("Even bin count supplied! Bin count raised by 1 to become odd.")
        num_bins += 1
    thresholds = pd.Series(np.linspace(max_delta * (-1), max_delta, num=(num_bins + 1)))
    shift = int((num_bins + 1) / 2)
    score = tuple(np.digitize(data_series, thresholds, right=False) - shift)
    return score


def state_list(series, state_func, att_num, days):
    att_list = state_func(series)  # convert input values to percentages

    data_list = [att_list[i - 1:i + days - 1] for i in range(1, len(att_list) - days)]
    # data_list is currently set to start at item 2 for percent change calculations

    data_series = pd.Series(data_list)  # list of lists of pre-portioned state data
    data_series = data_series * 100

    maxes = np.array([abs(percs).max() for percs in data_series])
    max_delta = maxes.max()  # maximum percent change defines the bin range for integerization
    int_changes = [integerize_state(data_series=pd.Series(state), num_bins=att_num, max_delta=max_delta)
                   for state in data_series]

    # state_history = [State(state).as_tuple() for state in int_changes]
    return int_changes

