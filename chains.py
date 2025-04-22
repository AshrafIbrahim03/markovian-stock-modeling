#! ./venv/bin/python
import tensorflow_probability as tfp
from get_state import get_state_by_percentile,get_state_by_zscore
from get_target import get_target
from abc import ABC, abstractmethod
import pandas as pd
from transition_fn import transition_fn_by_prob
from state import StateWindow
import datetime



class MarkovChain(ABC):
    
    @abstractmethod
    def set_current_state(self,new_state:tuple[int]):
        pass

    @abstractmethod
    def get_current_state(self):
        pass

    @abstractmethod
    def step(self,num:int):
        pass


class FirstMC(MarkovChain):
    current_state:str
    def __init__(self,initial_state:str):
        self.current_state = initial_state
    
    def step(self,num:int) -> list[str]:
        to_ret:list[str] = []
        for _ in range(num):
            next_state,_ = get_target(cur_state=self.current_state)
            to_ret.append(next_state)
            self.current_state = next_state
        return to_ret