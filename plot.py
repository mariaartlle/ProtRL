import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Font and line width settings
font = {'family': 'sans serif', 'size': 20}
plt.rc('font', **font)
plt.rcParams['lines.linewidth'] = 2.5

# Colors
colors = ['#37c6c8', '#c54f72', '#efefef', '#925aa6', '#c0416c', '#38c6c8']

# Load data
df = pd.read_csv('logs.csv')

# Plot
fig, ax = plt.subplots()
sns.lineplot(data=df, x='iteration_num', y='lenght', color=colors[3], ax=ax)

ax.grid(False)
plt.ylim(0,1)
ax.set_xlabel('Iteration')
ax.set_ylabel('Length')
ax.legend()
plt.tight_layout()
plt.savefig("lenght_plot.png")
