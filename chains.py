#! ./venv/bin/python
from get_state import (
    get_state_by_percentile,
    get_state_by_zscore,
    get_state_by_perc_change,
)
from get_target import get_target, get_target_max
from abc import ABC, abstractmethod
import pandas as pd
from transition_fn import t_table_generator_by_prob, transition_fn_by_randomized_vector,transition_fn_by_lin_reg
from state import StateValidator, State
import datetime


class MarkovChain(ABC):
    @abstractmethod
    def set_current_state(self, new_state: tuple[int]):
        pass

    @abstractmethod
    def get_current_state(self):
        pass

    @abstractmethod
    def step(self, num: int):
        pass


class FirstMC(MarkovChain):
    current_state: str

    def __init__(self, initial_state: str):
        self.current_state = initial_state

    def set_current_state(self, new_state: tuple[int]):
        assert len(new_state) == len(self.current_state)

        self.current_state = new_state

    def get_current_state(self):
        return self.current_state

    def step(self, num: int) -> list[str]:
        to_ret: list[str] = []
        for _ in range(num):
            next_state, _ = get_target(cur_state=self.current_state)
            to_ret.append(next_state)
            self.current_state = next_state
        return to_ret


class FeedingMCException(Exception):
    _error_type: int

    ILLEGAL_WINDOW_END = 1

    def __init__(self, err_type: int):
        assert err_type == FeedingMCException.ILLEGAL_WINDOW_END
        self._error_type = err_type

    def get_error_code(self) -> int:
        return self._error_type


class FeedingPercAggMC(MarkovChain):
    """This class uses get_target_by_perc_change, transition_fn_by_prob, and get_target for each step
    # `step` FUNCTION **NOT WORKING**
    """

    data_window_start: int
    data_window_end: int
    window_len: int
    data: pd.Series
    p_matrix: pd.DataFrame
    validator: StateValidator

    def __init__(
        self,
        data: pd.Series,
        window_len: int = 3,
        year_horizon: datetime.datetime = datetime.datetime(2015, 3, 1),
    ):
        self.data = data
        self.data_window_start = 0
        assert data.size > window_len
        assert window_len > 0
        self.data_window_end = window_len
        self.window_len = window_len
        self.validator = StateValidator(11, window_len)
        self.p_matrix = t_table_generator_by_prob(
            "./data/inflation_adjusted_berkshire_stocks.csv",
            year_horizon=year_horizon,
            days=window_len,
        )

    def set_current_state(self, new_state: State):
        assert len(new_state) == len(self.current_state)
        assert type(new_state) is State
        assert self.validator.is_valid_state(new_state)

        self.current_state = new_state

    def get_current_state(self):
        return self.current_state

    def _step(self) -> State:
        if self.data_window_end > self.data.size:
            raise FeedingMCException(FeedingMCException.ILLEGAL_WINDOW_END)
        window = self.data[self.data_window_start : self.data_window_end]
        assert window.size == self.window_len
        current_state = State(tuple(get_state_by_perc_change(window)))
        next_state, _ = get_target(self.p_matrix, current_state.as_tuple())
        self.data_window_start += 1
        self.data_window_end += 1
        return next_state

    def step(self, num: int) -> list[State]:
        to_ret = []
        for _ in range(num):
            to_ret.append(self._step())


class RandomizedMaxProbMC(MarkovChain):
    """This Markov Chain uses `transition_fn_by_randomized_vector` to generate probabilities then picks using"""

    current_state: State
    validator: StateValidator

    def __init__(self, initial_state: State, validator: StateValidator):
        assert validator.is_valid_state(initial_state)
        self.current_state = initial_state
        self.validator = validator

    def set_current_state(self, new_state: State):
        assert len(new_state) == len(self.current_state)

        self.current_state = new_state

    def get_current_state(self):
        return self.current_state

    def _step(self) -> State:
        probs = transition_fn_by_randomized_vector(self.current_state, self.validator)
        next_state = State(get_target_max(probs))
        self.current_state = next_state
        return next_state

    def step(self, num: int) -> list[State]:
        to_ret: list[State] = []
        for _ in range(num):
            to_ret.append(self._step())

        return to_ret
