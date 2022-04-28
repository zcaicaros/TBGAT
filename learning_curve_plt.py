import numpy as np
import matplotlib.pyplot as plt


j = 10  # 10， 15， 15， 20， 20
m = 10  # 10， 10， 15， 10， 15
batch_size = 64
episodes = 128000  # 128000, 256000
step_validation = 10

# plot parameters
total_plt_steps = 200
show = True
save = False
log_type = 'training'  # 'training', 'validation'
value_type = 'last_step'  # 'last_step', 'incumbent'
plot_step_size_training = (episodes // batch_size) // total_plt_steps
plot_step_size_validation = (episodes // batch_size) // (total_plt_steps * 10)
save_file_type = '.pdf'
x_label_scale = 17
y_label_scale = 17
anchor_text_size = 17


if log_type == 'training':
    file1 = './log/training_log_{}x{}.npy'.format(j, m)
    log1 = np.load(file1)
    obj1 = log1[:log1.shape[0] // plot_step_size_training * plot_step_size_training].reshape(log1.shape[0] // plot_step_size_training, -1).mean(axis=1)
    # plotting...
    plt.figure(figsize=(8, 5.5))
    plt.xlabel('Every {} batches.'.format(r'${}$'.format(plot_step_size_training)), {'size': x_label_scale})
    plt.ylabel('Training', {'size': y_label_scale})
    plt.grid()
    x = np.array([i + 1 for i in range(obj1.shape[0])])
    plt.plot(x, obj1, color='tab:blue', label='{}×{}'.format(10, 10))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('merged_training_log', save_file_type))
    if show:
        plt.show()

else:
    file1 = './log/validation_log_{}x{}.npy'.format(j, m)
    log1 = np.load(file1)
    obj_incumbent1 = log1[:log1.shape[0] // plot_step_size_validation * plot_step_size_validation, 0].reshape(log1.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    # plotting...
    # plt.xlabel('Iteration(stride-{})'.format(plot_step_size_validation), {'size': x_label_scale})
    plt.figure(figsize=(8, 5.5))
    plt.xlabel('Every {} batches.'.format(r'${}$'.format(plot_step_size_validation*10)), {'size': x_label_scale})
    plt.ylabel('Incumbent', {'size': y_label_scale})
    plt.grid()
    x1 = np.array([i + 1 for i in range(obj_incumbent1.shape[0])])
    plt.plot(x1, obj_incumbent1, color='tab:blue', label='{}×{}'.format(10, 10))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('merged_incumbent_validation_log', save_file_type))
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
    plt.plot(x1, obj_last_step1, color='tab:blue', label='{}×{}'.format(10, 10))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('merged_last-step_validation_log', save_file_type))
    if show:
        plt.show()


