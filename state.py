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
    
    def __str__(self):
        return "["+ " ".join(self.state_window) +"]"

class StateValidator:
    """
    Validator for states. Can check states to see if they adhere to its rules
    The bounds for states are defined by the number passed to `num_bins` in the constructor. See `__init__` for more details
    """
    num_bins:int

    def __init__(self,num_bins:int):
        """ Creates a StateValidator object

        Arguments
        num_bins(int) -> Total number of bins, e.g. if 11 is passed through,
            then that means that the value 0 will represent no change in value,
            and there will be 5 states representing positive change and 5 states representing negative change.
        """
        assert num_bins%2 ==1 # num_bins should be odd
        self.num_bins = num_bins


    def is_valid_state(self,window:StateWindow):
        """ Makes sure that a passed in state is valid given the StateValidator's implementation
        """
        max_state = (self.num_bins-1) /2
        states = tuple(map(lambda x: abs(x),window.as_tuple()))

        return all(map(lambda val: val <=max_state,states))

    def get_max_bin(self)->int:
        return (self.num_bins-1) /2