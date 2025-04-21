import pandas as pd
import numpy as np
import get_state
import datetime
import json


def get_target(t_table: pd.DataFrame,
               cur_state=(3, 2, 1), days_into_future: int = 1):
    """
    This function takes a transition table t_table and calculates the most probable state to occur in
    days_into_future days from the given cur_state.
    returns (result_state, result_prob)
    """
    # TODO: validate behavior for multiple transitions in future (make sure probs sum to 1)
    cur_state = tuple(int(x) for x in cur_state)
    matrix = t_table.fillna(0)
    power_matrix = matrix ** days_into_future  # calculate P matrix for multiple transitions in the future
    # use additional parentheses and comma to force composite index recognition
    print(type(power_matrix.index))
    print(cur_state in power_matrix.index)
    prob_list = power_matrix.loc[[cur_state]].squeeze()
    prob_list = prob_list.sort_values(ascending=False)
    result_state = prob_list.index[0]
    result_prob = prob_list.iloc[0]
    return result_state, result_prob


