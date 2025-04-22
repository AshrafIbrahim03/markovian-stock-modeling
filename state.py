
class StateWindow:
    """
        This class represents a state in a Markov Chain. It abstracts a tuple of integers. The most recent state is the last element in a tuple
    """
    state_window: tuple[int]
    def __init__(self,state_win: tuple[int]):
        self.state_window = state_win

    def __eq__(self, value) -> bool:
        assert type(value) == StateWindow
        assert len(self.state_window) == len(value.state_window)

        for (s1,s2) in zip(self.state_window,value.state_window):
            if s1 is not s2:
                return False
        return True
    def as_tuple(self) -> tuple[int]:
        return self.state_window
    
    def __str__(self):
        return "["+ " ".join(self.state_window) +"]"