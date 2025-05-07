import ast

import datetime as datetime
from matplotlib import pyplot as plt
import pandas as pd
import chains
import datetime
import numpy as np
from scipy.stats import linregress
from itertools import product

import get_state
from chains import FreqModelMC

df = pd.read_csv('freq_results1.csv')
df['predicted_state'] = df['predicted_state'].apply(ast.literal_eval)
df['non_neutral_rec'] = df['predicted_state'].apply(lambda tup: int(tup[-1] != 0))
df['adjusted_acc'] = df['is_prediction_correct'] / (0.185089974293059 * 2 / df['num_att'])

# add non_neutral binary column


# parameter grouping
att_groups = df.groupby(['num_att', 'win_length'])[['non_neutral_rec', 'is_prediction_correct', 'adjusted_acc']].agg(['mean', 'std'])
# att_groups = df.groupby('num_att')['adjusted_acc'].agg(['mean', 'std'])


# assorted error bars for parameters
# plt.errorbar(x=att_groups.index, y=att_groups['mean'], yerr=att_groups['std'], fmt='o', capsize=5)
# plt.title('Frequency model adjusted accuracy by attribute number')
# plt.ylabel('Adjusted accuracy')
# plt.xlabel('Attribute number')
# plt.show()


# accuracy and non-neutral prediction correlation
# scatter can be adjusted to show error if desired
plt.scatter(x=att_groups['non_neutral_rec']['mean'], y=att_groups['is_prediction_correct']['mean'])
plt.title('Frequency model % accuracy versus non-neutral predictions')
plt.ylabel('Percent accuracy')
plt.xlabel('Proportion of non-neutral predictions')

slope, intercept, r_value, p_value, std_err = linregress(att_groups['non_neutral_rec']['mean'], att_groups['is_prediction_correct']['mean'])
x_fit = np.linspace(att_groups['non_neutral_rec']['mean'].min(), att_groups['non_neutral_rec']['mean'].max(), 100)
y_fit = slope * x_fit + intercept
plt.plot(x_fit, y_fit, color='red', label=f'Line of best fit, slope = {slope}')
plt.show()
# print(r_value ** 2, p_value)
print(slope)
print(att_groups)


# actual percent change with shaded region for neutral bin
att_range = range(3, 15, 2)
days_range = range(3, 5, 2)
# pair = (3, 1)

df = pd.read_csv('data/inflation_adjusted_berkshire_stocks.csv')
df['Date'] = pd.to_datetime(df['Date'])
train_range = (datetime.datetime(2005, 1, 1), datetime.datetime(2021, 1, 1))
sample_range=(datetime.datetime(2020, 1, 2), datetime.datetime(2020, 12, 31))
df_cut = df[(df['Date'] >= sample_range[0]) & (df['Date'] <= sample_range[1])]


for pair in product(att_range, days_range):
    chain = FreqModelMC(df, train_range=(datetime.datetime(2005, 1, 1), datetime.datetime(2021, 1, 1)), att_num=pair[0], days=pair[1], sample_range=(datetime.datetime(2020, 1, 2), datetime.datetime(2020, 12, 31)))

# chain = chains.FreqModelMC(df, train_range=(datetime.datetime(2005, 1, 1), datetime.datetime(2021, 1, 1)), att_num=pair[0], days=pair[1], sample_range=(datetime.datetime(2020, 1, 2), datetime.datetime(2020, 12, 31)))
# 13    18.508997


# Neutral region graph
raw_df = pd.read_csv('./data/inflation_adjusted_berkshire_stocks.csv')
perc_changes = get_state.get_state_by_perc_change(raw_df['Open_adjusted'].astype(float))
print(perc_changes.abs().max())
# plt.plot(perc_changes)
# plt.title("Percent daily change")
# plt.xlabel('Day')
# plt.ylabel('Percent change')
# plt.axhspan(-.05657, .05657, color='blue', alpha=0.3, label='Shaded y')
# plt.show()

# histogram showing distribution of attributes

# df = pd.read_csv('data/inflation_adjusted_berkshire_stocks.csv')
# pair = (25, 1)
# chain = chains.FreqModelMC(df, train_range=(datetime.datetime(2005, 1, 1), datetime.datetime(2021, 1, 1)), att_num=pair[0], days=pair[1], sample_range=(datetime.datetime(2020, 1, 2), datetime.datetime(2020, 12, 31)))
# atts = chain.all_states
# flattened = [x[0] for x in atts]
# min = pd.Series(flattened).min()
# max = pd.Series(flattened).max()
#
# print(flattened)
# # bins=[-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
# plt.hist(flattened, bins=range(min, max + 2), density=True)
# plt.title(f'Frequency of attributes with attribute number {pair[0]}')
# plt.xticks(range(min, max + 1))
# plt.show()


#
# # Calculate the running correct percentage using a cumulative average
# running_percentage = res['is_prediction_correct'].expanding().mean()
#
# fig, ax1 = plt.subplots(figsize=(10, 6))
#
# # Plot running percentage on the left y-axis
# ax1.plot(running_percentage.index, running_percentage, label='Running Correct Percentage', color='blue')
# ax1.set_xlabel('Number of days run')
# ax1.set_ylabel('Cumulative Accuracy', color='blue')
# ax1.tick_params(axis='y', labelcolor='blue')
# ax1.set_ylim(0.6, 1)
# ax1.grid(True)
# ax1.set_title('Running Accuracy over Time with Daily Price Changes')
#
# # Overlay the price changes on a secondary y-axis (right side)
# ax2 = ax1.twinx()
# graphed_data = data[:7500]
# ax2.plot(graphed_data.index, graphed_data.rolling(90).std(), label='Daily Price Change', color='orange', alpha=0.5)
# ax2.set_ylabel('Price Change', color='orange')
# ax2.tick_params(axis='y', labelcolor='orange')
#
# # Add a combined legend
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
#
# plt.show()
