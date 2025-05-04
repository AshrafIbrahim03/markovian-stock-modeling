import pandas as pd
import datetime
import chains
from chains import MarkovChain, FreqModelMC
import transition_fn
import get_state
import get_target
import os

from state import State

df = pd.read_csv('./data/inflation_adjusted_berkshire_stocks.csv')
df = pd.Series(df['Open_adjusted'])
linreg = chains.LinRegMC(df, 3, num_bins=3)

class ChainEvaluator:
    """
    When supplied with a markov chain and a range of training days, this class represents a functioning prediction model.
    day_correct: returns a binary value representing a correct guess by the model
    set_params: Used to alter the parameters of the chain without discarding any progress recorded

    """
    results = pd.DataFrame(columns=['model', 'att_num', 'window_length', 'day', 'correct'])
    att_num = None
    window_length = None

    def __init__(self, chain:MarkovChain, days, data, sample_range):
        self.all_states = None
        self.chain = chain
        self.days = days
        self.data = data
        self.sample_range = sample_range

    def evaluate(self):
        assert isinstance(self.chain, MarkovChain)



    def set_params(self, num_atts, window_length):
        self.num_atts = num_atts
        self.window_length = window_length
        if isinstance(self.chain, FreqModelMC):
            self.chain.regen_p_matrix()

        cut_range = self.data[(self.sample_range[0] < self.data['Date']) & (self.data['Date'] <= self.sample_range[1])]
        # ENTIRE DATA USED, NO YEAR HORIZON

        cut_att_list = get_state.get_state_by_perc_change(cut_range['Open_adjusted'])
        whole_att_list = get_state.get_state_by_perc_change(self.data['Open_adjusted'])

        cut_data_series = pd.Series(cut_att_list) * 100  # list of lists of pre-portioned state data
        whole_data_series = pd.Series(whole_att_list) * 100
        max_delta = whole_data_series.max()

        # list of all attributes for checking correctness
        all_states = [
            get_state.integerize_state(data_series=pd.Series(cut_data_series), num_bins=self.att_num, max_delta=max_delta)]
        all_states = [all_states[i - 1:i + self.days - 1] for i in range(1, len(all_states) - self.days)]
        self.all_states = [State(i) for i in all_states]
        # !may need to change type here!

    def day_is_correct(self, day:datetime.datetime):
        if

    def regen_p_matrix(self):
        """
        Only for chains like Freq model that keep a running p matrix
        """
