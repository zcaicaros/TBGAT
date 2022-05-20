import numpy as np
import matplotlib.pyplot as plt


# env parameters
tabu_size = 20
# model parameters
hidden_channels = 128
out_channels = 128
dropout_for_gat = 0
# training parameters
j = 10
m = 10
lr = 1e-5
steps_learn = 10
transit = 500
batch_size = 64
total_instances = 64000
step_validation = 10
ent_coeff = 1e-5
embed_tabu_label = False

algo_config1 = '{}_{}-{}-{}-{}_{}x{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
    # env parameters
    tabu_size,
    # model parameters
    hidden_channels, out_channels, 4, dropout_for_gat,
    # training parameters
    j, m, lr, steps_learn, transit, batch_size, 32000, step_validation, ent_coeff, embed_tabu_label
)

algo_config2 = '{}_{}-{}-{}-{}_{}x{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
    # env parameters
    tabu_size,
    # model parameters
    hidden_channels, out_channels, 8, dropout_for_gat,
    # training parameters
    j, m, lr, steps_learn, transit, batch_size, 32000, step_validation, ent_coeff, True
)

# plot parameters
total_plt_steps = 100
show = True
save = True
plot_step_size_training = (total_instances // batch_size) // total_plt_steps
save_file_type = '.pdf'
x_label_scale = 17
y_label_scale = 17
anchor_text_size = 17

file1 = './log/training_log_' + algo_config1 + '.npy'
log1 = np.load(file1)
file2 = './log/training_log_' + algo_config2 + '.npy'
log2 = np.load(file2)
obj1 = log1[:log1.shape[0] // plot_step_size_training * plot_step_size_training].reshape(log1.shape[0] // plot_step_size_training, -1).mean(axis=1)
obj2 = log2[:log2.shape[0] // plot_step_size_training * plot_step_size_training].reshape(log2.shape[0] // plot_step_size_training, -1).mean(axis=1)
# plotting...
plt.figure(figsize=(8, 5.5))
plt.xlabel('Every {} batches.'.format(r'${}$'.format(plot_step_size_training)), {'size': x_label_scale})
plt.ylabel('Cmax', {'size': y_label_scale})
plt.grid()
x = np.array([i + 1 for i in range(obj1.shape[0])])
plt.plot(x, obj1, color='tab:blue', label='{}×{} 4-heads'.format(j, m))
plt.plot(x, obj2, color='#f19b61', label='{}×{} 8-heads'.format(j, m))
plt.tight_layout()
plt.legend(fontsize=anchor_text_size)
if save:
    plt.savefig('./{}{}'.format('number_of_heads', save_file_type))
if show:
    plt.show()



