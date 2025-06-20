import numpy as np, matplotlib.pyplot as plt, seaborn as sns, os, glob

root_dir = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(root_dir, "data", "results")
label = "MDEMKAYVAL"

files = sorted(glob.glob(os.path.join(folder, f"len_stats_{label}_iteration_*.npy")))
means, stds = zip(*(np.load(f) for f in files))
iters = np.arange(len(files))

sns.lineplot(x=iters, y=means)
plt.fill_between(iters, np.array(means) - stds, np.array(means) + stds, alpha=.3)
plt.xlabel("Iteration"); plt.ylabel("Length"); plt.tight_layout()
plt.savefig("len_results.png", dpi=300)
