from itertools import combinations_with_replacement
class StateWindow:
    """
        This class represents a state in a Markov Chain. It abstracts a tuple of integers. The most recent state is the last element in a tuple
    """
    state_window: tuple[int]
    def __init__(self,state_win: tuple[int]):
        self.state_window = state_win

    def __eq__(self, value) -> bool:
        assert type(value) == StateWindow
        assert len(self.state_window) == len(value.state_window)# Two windows must have the same length to even see if they're equal

        for (s1,s2) in zip(self.state_window,value.state_window):
            if s1 is not s2:
                return False
        return True
    def as_tuple(self) -> tuple[int]:
        return self.state_window
    
    def get_next_state(self,next_bin:int):
        """Given a bin, returns a StateWindow object with the given bin as the most recent state

        Returns:
        StateWindow
        """
        assert type(next_bin) == int
        t = self.as_tuple()
        t = t + tuple([next_bin])
        return StateWindow(t[1:])
    
    def __len__(self)->int:
        return len(self.state_window)

    def __str__(self):
        return "["+ " ".join(self.state_window) +"]"


class ValidationException(Exception):
    """Wrapper class to define different error types
    """
    LEGAL_STATE:int = 0 # The passed state does not violate any rules from the StateValidator
    ILLEGAL_BIN:int=1 # One of the bins in the StateWindow that was passed is not in the bounds defined by the Validator
    ILLEGAL_STATE_LEN:int=2 # The StateWindow that was passed has the wrong length
    ILLEGAL_NEXT_STATE:int=3 # The next state passed cannot happen after the current state passed

    error_type:int

    def __init__(self,err_type:int, *args):
        assert err_type == ValidationException.ILLEGAL_BIN or err_type == ValidationException.ILLEGAL_STATE_LEN
        self.error_type = err_type
        super().__init__(*args)


class StateValidator:
    """
    Validator for states. Can check states to see if they adhere to its rules
    The bounds for states are defined by the number passed to `num_bins` in the constructor. See `__init__` for more details
    """
    num_bins:int
    window_len:int

    def __init__(self,num_bins:int,window_len:int):
        """ Creates a StateValidator object

        Arguments
        num_bins(int) -> Total number of bins, e.g. if 11 is passed through,
            then that means that the value 0 will represent no change in value,
            and there will be 5 states representing positive change and 5 states representing negative change.
        """
        assert num_bins%2 ==1 # num_bins should be odd
        assert window_len>0
        self.num_bins = num_bins
        self.window_len = window_len


    def is_valid_state(self,window:StateWindow)-> int:
        """ Makes sure that a passed in state is valid given the StateValidator's implementation

        Arguments:
        window(StateWindow) -> The state window to be checked

        Returns:
        An int based on what is not valid. Refer to `ValidationException` for more details.
        """
        assert type(window) == StateWindow
        max_state = (self.num_bins-1) /2
        states = tuple(map(lambda x: abs(x),window.as_tuple()))
        if not all(map(lambda val: val <=max_state,states)):
            return ValidationException.ILLEGAL_BIN
        elif len(window) == self.window_len:
            return ValidationException.ILLEGAL_STATE_LEN
        
        return ValidationException.LEGAL_STATE
    
    def is_next_state_valid(self,current_state:StateWindow,next_state:StateWindow) -> int:
        """ Makes sure that the `next_state` passed can actually happen after the `current_state`

        Arguments:
        current_state(StateWindow) -> The current state window
        next_state(StateWindow) -> The next state window
        
        Returns:
        An integer that says what type of error occurred. 0 means that it's a legal state. Refer to `ValidationException` for more details.
        """
        if type(current_state) != StateWindow:
            current_state = StateWindow(current_state)
        if type(next_state) != StateWindow:
            next_state = StateWindow(next_state)

        if self.is_valid_state(current_state) != 0:
            return self.is_valid_state(current_state)
        elif self.is_valid_state(next_state) != 0:
            return self.is_valid_state(next_state)

        current_window = current_state.as_tuple()
        next_window = next_state.as_tuple()
        n = self.window_len

        # Checks to see that the last n elements of current_state are equal to the first n elements of next_state
        for (current,next) in zip(current_window[1:],next_window[:n-1]):
            if current != next:
                return ValidationException.ILLEGAL_NEXT_STATE
        return ValidationException.LEGAL_STATE



        
        

    def get_max_bin(self)->int:
        return int((self.num_bins-1) /2)
    
    def gen_state_space(self):
        max_bin = self.get_max_bin()
        assert type(max_bin) == int
        return combinations_with_replacement(range(-1*max_bin,max_bin+1),self.window_len)
