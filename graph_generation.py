from matplotlib import pyplot as plt
import pandas as pd

# Calculate the running correct percentage using a cumulative average
running_percentage = res['is_prediction_correct'].expanding().mean()

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot running percentage on the left y-axis
ax1.plot(running_percentage.index, running_percentage, label='Running Correct Percentage', color='blue')
ax1.set_xlabel('Number of days run')
ax1.set_ylabel('Cumulative Accuracy', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(0.6, 1)
ax1.grid(True)
ax1.set_title('Running Accuracy over Time with Daily Price Changes')

# Overlay the price changes on a secondary y-axis (right side)
ax2 = ax1.twinx()
graphed_data = data[:7500]
ax2.plot(graphed_data.index, graphed_data.rolling(90).std(), label='Daily Price Change', color='orange', alpha=0.5)
ax2.set_ylabel('Price Change', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Add a combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.show()
