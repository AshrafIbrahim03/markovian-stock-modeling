import pandas as pd
import numpy as np
import get_state
import datetime
import time
from collections import Counter
import json
from state import State,StateValidator,ValidationException
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import scipy.stats as st

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
                 num_bins: int = 20,
                 max_delta: int = 20):
        # key = {np.linspace(-100, 100, num=num_bins)[i]: i for i in range(num_bins)}
        thresholds = pd.Series(np.linspace(max_delta * (-1), max_delta, num=num_bins))
        self.combination = tuple(np.digitize(series, thresholds, right=False))

    def get_combination(self):
        return self.combination

    def __str__(self):
        return str(self.combination)


# For testing purposes, it is ok to use the below transition function with all default args
def t_table_generator_by_prob(path: str,
                          year_horizon: datetime = datetime.datetime(2015, 1, 1),
                          today: datetime = datetime.datetime(2025, 3, 1),
                          col: str = 'Open_adjusted',
                          days: int = 3,
                          state_func=get_state.get_state_by_perc_change
                          ) -> []:
    """
    This function returns a transition probability matrix based on the frequency of past transitions.

    year_horizon: earliest year to include in assessment of transition history
    path: path to csv file containing numeric values
    col: string name of column within the csv located at path that contains the desired values
    days: number of days included in each state; aka "state length"
    """
    # print('pre-read: ' + str(time.time()))
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(year_horizon < df['Date']) & (df['Date'] <= today)]

    # print('pre-data_series: ' + str(time.time()))
    data_list = [df[col][i - 1:i + days] for i in range(1, len(df) - days)]
    # data_list is currently set to start at item 2 for percent change calculations

    data_series = pd.Series(data_list)

    # print('pre-state: ' + str(time.time()))
    perc_changes = [state_func(data_series=state, state_width=1, min=state.min(), max=state.max()) for state in data_series]
    maxes = np.array([abs(percs).max() for percs in perc_changes])
    max_delta = maxes.max()
    # print('max delta')
    # print(max_delta)

    state_history = [Combination(state_func
                                 (data_series=state, state_width=1, min=state.min(),
                                  max=state.max()), max_delta=max_delta).get_combination() for state in data_series]
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
    # print('post zip: ' + str(time.time()))

    data = [[pairs.count((i, j)) / counts[i] if counts[i] > 0 else 0
             for j in combos]
            for i in combos]
    result_df = pd.DataFrame(data, index=combos, columns=combos)
    # result_df.to_csv('transition_table.csv')
    return result_df

def transition_fn_by_p_matrix(p_matrix:pd.DataFrame,current_state:tuple[int],steps:int=1)->pd.Series:
    """ Takes in a p_matrix table and the number of steps into the future to predict
    Arguments:
    p_matrix(pd.Series) ->  This is a dataframe where the index should be equal to the columns . Each row sums to one.
    current_state(tuple[int]) -> A tuple of ints that should be in the p_matrix index
    steps(int) -> The number of steps into the future to predict. 

    Returns:
    A series whose indices are the different states and the values are the probability of that state occurring
    """
    assert p_matrix.index.isin(p_matrix.columns).all() # the Index should be equal to the columns 
    assert p_matrix.apply(lambda row: row.sum() ==1).all() # Each row should sum to 1
    assert p_matrix.index.isin([current_state]).any() # current_state should be in the index
    p_mat = p_matrix ** steps

    return p_mat.T[current_state]

def transition_fn_by_randomized_vector(current_state:State,validator:StateValidator)-> pd.Series:
    """Returns a Series whose indices are all the possible states and randomized probabilities assigned to each one"""
    all_states = list(validator.gen_state_space())
    valid_states = [state for state in all_states if validator.is_next_state_valid(current_state,state)]
    num_valid_states = len(valid_states)

    rand_values = np.random.rand(num_valid_states)
    rand_values /= rand_values.sum()
    final_series = pd.concat([
        pd.Series(rand_values, index=valid_states),
        pd.Series(0, index=[state for state in all_states if state not in valid_states])
    ])
    return final_series


    # prob_dict = {state: 0 for state in all_states}
    # for state, prob in zip(valid_states, rand_values):
    #     prob_dict[state] = prob

    # return pd.Series(prob_dict)



def transition_fn_by_lin_reg(current_state:State,validator:StateValidator) -> pd.Series:
    """ Computes the line of best fit over the current states and outputs the probabilities of
        next states based on distance from the expected value

        Returns a pd.Series with the states as the indices and the probabilities as values
    """
    if isinstance(current_state,State):
        X:tuple[int] = current_state.as_tuple()
    elif isinstance(current_state, tuple):
        X = current_state
        current_state = State(current_state)
    else:
        raise Exception("current_state must be State or tuple")
    Y = np.arange(len(X))
    # slope,intercept = np.polyfit(X,Y,1)
    slope,intercept = np.polynomial.polynomial.Polynomial.fit(X,Y,deg=1)
    # next_state has to be an int that falls within the ranges enforced by validator
    next_state_int = int((slope * len(X)) + intercept)
    next_state_int = int(np.clip(next_state_int,validator.get_min_attr(),validator.get_max_attr()))

    mean = np.mean(current_state.as_tuple())

    std = np.std(current_state.as_tuple())
    if std == 0:# A stdev of 0 makes it so that all the probs are NaN so another value should be swapped in
        stdevs = map(np.std,validator.gen_state_space())
        non_zero_stdevs = filter(lambda x: x!=0,stdevs)
        min_std = min(non_zero_stdevs) / 2
        std = min_std
    bins = np.arange(validator.get_min_attr() - 0.5, validator.get_max_attr() + 1, 1)
    bins[0] = -1* np.inf
    bins[-1] = np.inf
    cdf_s = st.norm.cdf(x=bins,loc=mean,scale=std)
    states = np.arange(validator.get_min_attr(),validator.get_max_attr()+1,1)

    bin_probs = [cdf_s[i+1] - cdf_s[i] for (i,_) in enumerate(cdf_s[:-1])]

    

    return pd.Series(bin_probs,index=states)

def t_table_gen_by_lin_reg_og(validator:StateValidator) -> pd.DataFrame:
    to_ret:pd.DataFrame = pd.DataFrame()
    # Create a comprehensive list of all state keys as strings from the state space
    complete_index = [str(generated) for generated in validator.gen_state_space()]
    for state in validator.gen_state_space():
        prob_vec = transition_fn_by_lin_reg(state,validator)
        new_index = [str((*state[1:], n)) for n in prob_vec.index]
        prob_vec.index = new_index
        # Reindex the Series to ensure it has all keys, filling missing values with 0
        prob_vec = prob_vec.reindex(complete_index, fill_value=0)
        assert np.isclose(np.sum(prob_vec),1), f"prob_vec for {state} is not 1 but is {np.sum(prob_vec)}"
        print(np.sum(prob_vec))
        to_ret[str(state)] = prob_vec
    
    return to_ret.T

def t_table_gen_by_lin_reg(validator:StateValidator) -> pd.DataFrame:
    """
    DOES NOT WORK YET
    """
    to_ret:pd.DataFrame = pd.DataFrame()
    # Create a comprehensive list of all state keys as strings from the state space
    complete_index = [str(generated) for generated in validator.gen_state_space()]
    for state in validator.gen_state_space():
        zeroes = pd.Series(np.zeros(len(complete_index)),index=complete_index)
        prob_vec = transition_fn_by_lin_reg(state,validator)
        new_index = [str((*state[1:], n)) for n in prob_vec.index]
        prob_vec.index = new_index

        

        # # Reindex the Series to ensure it has all keys, filling missing values with 0
        # prob_vec = prob_vec.reindex(complete_index, fill_value=0)
        assert np.isclose(np.sum(prob_vec),1), f"prob_vec for {state} is not 1 but is {np.sum(prob_vec)}"
        print(np.sum(prob_vec))
        to_ret[str(state)] = prob_vec
    
    return to_ret.T