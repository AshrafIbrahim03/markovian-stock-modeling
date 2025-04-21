import pandas as pd
import datetime
import transition_fn
import get_state
import get_target
import os

start = datetime.datetime(2021, 9, 28)
end = datetime.datetime(2021, 11, 30)
date_list = pd.date_range(start, end).to_list()
year_horizon = datetime.datetime(2015, 1, 1)
# TODO: Change interaction between transition_fn and get_state so the current state
#  can be generated with less repetition
df = pd.read_csv('/Users/walkerwatson/PycharmProjects/markovian-stock-modeling/data/inflation_adjusted_berkshire_stocks.csv')
result = pd.DataFrame(columns=['date', 'cur_state', 'predicted', 'prob', 'actual', 'success'])
current_dir = os.path.dirname(__file__)
root = os.path.abspath(os.path.join(current_dir, '..'))
data_path = os.path.join(root, 'markovian-stock-modeling', 'data', 'inflation_adjusted_berkshire_stocks.csv')
for today in date_list:
    t_func = transition_fn.transition_fn_by_prob(today=today, path=data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(year_horizon < df['Date'])]
    col = 'Open_adjusted'
    # print('pre-data_series: ' + str(time.time()))
    today_index = df[df['Date'] == today].index
    if today_index.empty:
        print('date is empty!')
        continue
    pos = df.index.get_loc(today_index[0])
    data_series = df[col].iloc[pos - 3: pos + 1]

    # This part is repeated from transition_fn, which is bad
    today_state = transition_fn.Combination(get_state.get_state_by_perc_change
                                 (data_series=data_series, state_width=1), max_delta=8).get_combination()
    print(today_state)
    tom_data_series = df[col].iloc[pos - 2: pos + 2]
    tomorrow_state = transition_fn.Combination(get_state.get_state_by_perc_change
                                 (data_series=tom_data_series, state_width=1), max_delta=8).get_combination()
    print(tomorrow_state)
    predicted, prob = get_target.get_target(t_table=t_func, cur_state=today_state, days_into_future=1)

    result.loc[len(result)] = [today, today_state, predicted, prob, tomorrow_state, int(predicted == tomorrow_state)]

result.to_csv('result_sheet20.csv')
