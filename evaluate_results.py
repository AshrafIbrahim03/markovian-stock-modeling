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
from itertools import product

from state import State,StateValidator


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
        to_ret = pd.DataFrame(columns=["run", "num_att", 'win_length', "predicted_state", "actual_state", "is_prediction_correct"])
        if isinstance(chain, FreqModelMC):
            rows = chain.step(num_days)
            to_ret = pd.DataFrame(data={"predicted_state": rows[0], "actual_state": rows[1]})
            to_ret["is_prediction_correct"] = (to_ret['actual_state'] == to_ret['predicted_state']).astype(int)
        else:
            to_ret["predicted_state"] = [state[0].as_tuple() for state in chain.step(num_days)]
            next(self.buffer_reader) # need to start on the next window because the markov chain predicts what the next window will be
            raw_windows = (next(self.buffer_reader) for _ in range(num_days))
            to_ret["actual_state"] = [tuple(chain.get_states(window)) for window in raw_windows]
            to_ret["is_prediction_correct"] = to_ret['actual_state'].apply(tuple) == to_ret['predicted_state'].apply(tuple)
        to_ret['num_att'] = chain.att_num
        to_ret['win_length'] = chain.days
        return to_ret.dropna()
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
att_range = range(3, 15, 2)
days_range = range(3, 15, 2)

if os.path.exists('./freq_results1.csv'):
    final_df = pd.read_csv('freq_results1.csv')
else:
    final_df = pd.DataFrame(columns=['predicted_state', 'actual_state', 'is_prediction_correct', 'num_att', 'win_length'])

for pair in product(att_range, days_range):
    if ((final_df['num_att'] == pair[0]) & (final_df['win_length'] == pair[1])).any():
        print(f'Parameter combo num_att: {pair[0]} and win_length: {pair[1]} has already been covered. Moving on.')
        continue
    chain = FreqModelMC(df, train_range=(datetime.datetime(2005, 1, 1), datetime.datetime(2021, 1, 1)), att_num=pair[0], days=pair[1], sample_range=(datetime.datetime(2020, 1, 2), datetime.datetime(2020, 12, 31)))
    result = eval.evaluate_as_df(chain, 1000)
    final_df = pd.concat([final_df, result])
    final_df.to_csv('freq_results1.csv')
    print(final_df)

