import numpy as np
import matplotlib.pyplot as plt


# env parameters
tabu_size = 20
# model parameters
hidden_channels = 128
out_channels = 128
heads = 4
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

algo_config = '{}_{}-{}-{}-{}_{}x{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
    # env parameters
    tabu_size,
    # model parameters
    hidden_channels, out_channels, heads, dropout_for_gat,
    # training parameters
    j, m, lr, steps_learn, transit, batch_size, total_instances, step_validation, ent_coeff, embed_tabu_label
)

# plot parameters
total_plt_steps = 50
show = True
save = False
log_type = 'validation'  # 'training', 'validation'
value_type = 'last_step'  # 'last_step', 'incumbent'
plot_step_size_training = (total_instances // batch_size) // total_plt_steps
plot_step_size_validation = (total_instances // batch_size) // (total_plt_steps * 10)
save_file_type = '.pdf'
x_label_scale = 17
y_label_scale = 17
anchor_text_size = 17


if log_type == 'training':
    file1 = './log/training_log_' + algo_config + '.npy'
    log1 = np.load(file1)
    obj1 = log1[:log1.shape[0] // plot_step_size_training * plot_step_size_training].reshape(log1.shape[0] // plot_step_size_training, -1).mean(axis=1)
    # plotting...
    plt.figure(figsize=(8, 5.5))
    plt.xlabel('Every {} batches.'.format(r'${}$'.format(plot_step_size_training)), {'size': x_label_scale})
    plt.ylabel('Cmax', {'size': y_label_scale})
    plt.grid()
    x = np.array([i + 1 for i in range(obj1.shape[0])])
    plt.plot(x, obj1, color='tab:blue', label='{}×{}'.format(j, m))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('training_curve', save_file_type))
    if show:
        plt.show()

else:
    file1 = './log/validation_log_' + algo_config + '.npy'
    log1 = np.load(file1)
    obj_incumbent1 = log1[:log1.shape[0] // plot_step_size_validation * plot_step_size_validation, 0].reshape(log1.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    # plotting...
    # plt.xlabel('Iteration(stride-{})'.format(plot_step_size_validation), {'size': x_label_scale})
    plt.figure(figsize=(8, 5.5))
    plt.xlabel('Every {} batches.'.format(r'${}$'.format(plot_step_size_validation*10)), {'size': x_label_scale})
    plt.ylabel('Incumbent', {'size': y_label_scale})
    plt.grid()
    x1 = np.array([i + 1 for i in range(obj_incumbent1.shape[0])])
    plt.plot(x1, obj_incumbent1, color='tab:blue', label='{}×{}'.format(j, m))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('validation_curve_incumbent', save_file_type))
    if show:
        plt.show()

    obj_last_step1 = log1[:log1.shape[0] // plot_step_size_validation * plot_step_size_validation, 1].reshape(log1.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    # plotting...
    # plt.xlabel('Iteration(stride-{})'.format(plot_step_size_validation), {'size': x_label_scale})
    plt.figure(figsize=(8, 5.5))
    plt.xlabel('Every {} batches.'.format(r'${}$'.format(plot_step_size_validation*10)), {'size': x_label_scale})
    plt.ylabel('Last-Step', {'size': y_label_scale})
    plt.grid()
    x1 = np.array([i + 1 for i in range(obj_incumbent1.shape[0])])
    plt.plot(x1, obj_last_step1, color='tab:blue', label='{}×{}'.format(j, m))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('validation_curve_last_step', save_file_type))
    if show:
        plt.show()


