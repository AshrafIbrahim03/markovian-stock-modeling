import pandas as pd

class BufferReader:
    data:pd.Series
    window_size:int
    index:int
    def __init__(self,data:pd.Series,window_size:int):
        assert data.size > window_size
        assert window_size> 0
        self.data = data
        self.index = 0
        self.window_size = window_size
    
    def __next__(self)->tuple[int]:
        end_index:int = self.index+self.window_size
        if end_index > self.data.size:
            raise StopIteration()
        current_window = self.data[self.index:end_index]
        self.index+=1
        return current_window
    
