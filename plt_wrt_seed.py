import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

train_log_seed_1 = np.load(
    'log/training_log_-1-fdd-divide-wkr_128-128-4-0-TPMCAM_10x10-5e-05-10-500-64-128000-10-1e-05-1-False-ls.npy'
)
train_log_seed_3 = np.load(
    'log/training_log_-1-fdd-divide-wkr_128-128-4-0-TPMCAM_10x10-5e-05-10-500-64-128000-10-1e-05-3-False-ls.npy'
)
train_log_seed_6 = np.load(
    'log/training_log_-1-fdd-divide-wkr_128-128-4-0-TPMCAM_10x10-5e-05-10-500-64-128000-10-1e-05-6-False-ls.npy'
)

mean = np.stack([
    train_log_seed_1,
    train_log_seed_3,
    train_log_seed_6
]).mean(axis=0)
std = np.stack([
    train_log_seed_1,
    train_log_seed_3,
    train_log_seed_6
]).std(axis=0)

scale = 4
total_plt_steps = 200
plot_step_size_training = (128000 // 64) // total_plt_steps
# plot_step_size_validation = (128000 // 64) // (total_plt_steps * 10)
fig, ax = plt.subplots(figsize=(scale * 1.618, scale))
x = np.arange(mean.shape[0] // plot_step_size_training)
mean = mean.reshape(mean.shape[0] // plot_step_size_training, -1).mean(-1)
std = std.reshape(std.shape[0] // plot_step_size_training, -1).mean(-1)

base_line1, = ax.plot(x, mean, 'blue', label="10x10")
ax.fill_between(x, mean - std / 2, mean + std / 2, facecolor=base_line1.get_color(), alpha=0.2)

ax.legend(fontsize=12, loc='upper right')
ax.xaxis.set_major_formatter(ticker.EngFormatter())
# ax.set(ylabel='Average Return')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# ax.set(xlabel='Episode')
# axs.get_xaxis().set_visible(False)
ax.yaxis.label.set_size(15)
ax.grid(True)
# ax.set_xticklabels([])
ax.xaxis.label.set_size(15)

plt.xlabel('Every {} batches.'.format(r'${}$'.format(plot_step_size_training)), {'size': 17})
plt.ylabel('Cmax', {'size': 17})

plt.tight_layout()
plt.savefig('training_curve_seed.pdf')
plt.show()
