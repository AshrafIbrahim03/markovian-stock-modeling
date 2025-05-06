import pandas as pd
import numpy as np
import datetime
import chains
from chains import MarkovChain, FreqModelMC
import transition_fn
import get_state
import get_target
import os
from buffer_reader import BufferReader

from state import State,StateValidator

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
        # if
        pass

    def regen_p_matrix(self):
        """
        Only for chains like Freq model that keep a running p matrix
        """


class Evaluator():
    
    chain:MarkovChain
    buffer_reader:BufferReader
    val:StateValidator
    def __init__(self,data:pd.Series,val:StateValidator):
        """Instantiates Evaluator object

        Args:
            data (pd.Series): The data to compare against
        """
        self.val = val
        self.buffer_reader = BufferReader(data,self.val.window_len)

    def evaluate_as_df(self,chain:MarkovChain,num_days:int)->pd.DataFrame:
        """Evaluates the MarkovChain

        Args:
            num_days (int): The number of days to evaluate over
            chain (MarkovChain): The MarkovChain to be evaluated. We can't store this as a class variable because we can't easily restart the MarkovChain

        Returns:
            pd.DataFrame: A dataframe with columns: ["predicted_state","actual_state","is_prediction_correct"]. predicted_state will be tuple[int], actual_state will be tuple[int], is_prediction_correct is a boolean representing if the prediction is correct
        """
        to_ret = pd.DataFrame(columns=["run","predicted_state","actual_state","is_prediction_correct"])
        to_ret["predicted_state"] = [state.as_tuple() for state in chain.step(num_days)]
        next(self.buffer_reader) # need to start on the next window because the markov chain predicts what the next window will be
        raw_windows = (next(self.buffer_reader) for _ in range(num_days))
        to_ret["actual_state"] = [tuple(chain.get_states(window)) for window in raw_windows]
        to_ret["is_prediction_correct"] = to_ret['actual_state'].apply(tuple) == to_ret['predicted_state'].apply(tuple)
        
        return to_ret
    def get_perc_correct(self,chain:MarkovChain,num_days:int)-> float:
        next(self.buffer_reader) # need to start on the next window because the markov chain predicts what the next window will be
        raw_windows = (next(self.buffer_reader) for _ in range(num_days))
        actual_states = (tuple(chain.get_states(window)) for window in raw_windows)
        predicted_states = (state.as_tuple() for state in chain.step(num_days))

        num_correct = 0
        num_iterated = 0
        for (actual,pred) in zip(actual_states,predicted_states):
            if tuple(actual) == tuple(pred):
                num_correct+=1
            num_iterated+=1
        
        return num_correct / num_iterated
            
df = pd.read_csv('data/inflation_adjusted_berkshire_stocks.csv')
data = df['Open_adjusted']

validator = StateValidator(3, 3)
eval = Evaluator(data=data, val=validator)
init_state = State(tuple([0, 0, 0]))
chain = FreqModelMC(init_state, df, datetime.datetime(2015, 1, 1), datetime.datetime(2025, 3, 1), 3, 3, sample_range=())
eval.get_perc_correct()