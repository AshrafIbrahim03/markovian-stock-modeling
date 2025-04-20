import pandas as pd
import numpy as np





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
    scores = np.clip(z_scores,-(states/2 * state_width),(states/2 * state_width))
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
    scores = np.clip(percentiles - 50,-50,50)

    return  np.round(scores / state_width) * state_width

def get_state_by_perc_change(data_series:pd.Series, states:int=6, state_width:float=1,min:float=0,max:float=100):
    """

        """
    start_days = data_series[:-1]
    start_days.reset_index(drop=True)
    pairs = list(zip(start_days, data_series[1:]))
    diffs = [p[1] - p[0] for p in pairs]

    perc_change = (diffs[i] / start_days.iloc[i] for i in range(len(start_days)))
    perc_change = pd.Series(perc_change)
    score = perc_change * 100
    print(score)
    return score


