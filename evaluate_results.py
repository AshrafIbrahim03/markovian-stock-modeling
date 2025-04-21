import pandas as pd
import datetime
import transition_fn
import get_state

start = datetime.datetime(2021, 9, 28)
end = datetime.datetime(2021, 12, 1)
date_list = pd.date_range(start, end).to_list()
year_horizon = datetime.datetime(2015, 1, 1)
# TODO: Change interaction between transition_fn and get_state so the current state
#  can be generated with less repetition
df = pd.read_csv('/Users/walkerwatson/PycharmProjects/markovian-stock-modeling/data/inflation_adjusted_berkshire_stocks.csv')
for today in date_list:
    t_func = transition_fn.transition_fn_by_prob(today=today)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(year_horizon < df['Date']) & (df['Date'] <= today)]
    col = 'Open_adjusted'
    # print('pre-data_series: ' + str(time.time()))
    today_index = df[df['Date'] == today].index
    if today_index.empty:
        continue
    data_list = df[col][today_index[0] - 3: today_index[0]]
    data_series = pd.Series(data_list)
    # This part is repeated from transition_fn, which is bad
    print(data_series)
    today_state = transition_fn.Combination(get_state.get_state_by_perc_change
                                 (data_series=data_series, state_width=1), max_delta=8).get_combination()
    print(today_state)
