import pandas as pd
import numpy as np
import get_state
import datetime
import json


def get_target(t_table: str = 'transition_table.json',
               cur_state: str = '[3 2 1]', days_into_future: int = 1):
    """
    This function takes a transition table t_table and calculates the most probable state to occur in
    days_into_future days from the given cur_state.
    returns (result_state, result_prob)
    """
    # TODO: validate behavior for multiple transitions in future (make sure probs sum to 1)

    df = pd.read_json(t_table)
    matrix = df.fillna(0)
    power_matrix = matrix ** days_into_future  # calculate P matrix for multiple transitions in the future
    prob_list = power_matrix[cur_state]
    prob_list = prob_list.sort_values(ascending=False)
    result_state = prob_list.index[0]
    result_prob = prob_list.iloc[0]
    return result_state, result_prob


print(get_target(days_into_future=5))
