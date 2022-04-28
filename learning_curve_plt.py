import numpy as np
import matplotlib.pyplot as plt


j = 20  # 10， 15， 15， 20， 20
m = 15  # 10， 10， 15， 10， 15
l = 1
h = 99
init_type = 'fdd-divide-mwkr'
reward_type = 'yaoxin'  # 'yaoxin', 'consecutive'
gamma = 1

hidden_dim = 128
embedding_layer = 4
policy_layer = 4
embedding_type = 'gin+dghan'  # 'gin', 'dghan', 'gin+dghan'
heads = 1
drop_out = 0.

lr = 5e-5  # 5e-5, 4e-5
steps_learn = 10
training_episode_length = 500
batch_size = 64
episodes = 128000  # 128000, 256000
step_validation = 10

# plot parameters
total_plt_steps = 200
show = True
save = False
log_type = 'validation'  # 'training', 'validation'
value_type = 'last_step'  # 'last_step', 'incumbent'
plot_step_size_training = (episodes // batch_size) // total_plt_steps
plot_step_size_validation = (episodes // batch_size) // (total_plt_steps * 10)
save_file_type = '.pdf'
x_label_scale = 17
y_label_scale = 17
anchor_text_size = 17


if embedding_type == 'gin':
    dghan_param_for_saved_model = 'NAN'
elif embedding_type == 'dghan' or embedding_type == 'gin+dghan':
    dghan_param_for_saved_model = '{}_{}'.format(heads, drop_out)
else:
    raise Exception('embedding_type should be one of "gin", "dghan", or "gin+dghan".')


file1 = 'validation_log.npy'
log1 = np.load(file1)


if log_type == 'training':
    obj1 = log1[:log1.shape[0] // plot_step_size_training * plot_step_size_training].reshape(log1.shape[0] // plot_step_size_training, -1).mean(axis=1)
    # plotting...
    plt.figure(figsize=(8, 5.5))
    plt.xlabel('Every {} batches.'.format(r'${}$'.format(plot_step_size_training)), {'size': x_label_scale})
    plt.ylabel('Gap to CP-SAT', {'size': y_label_scale})
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
    obj_incumbent1 = log1[:log1.shape[0] // plot_step_size_validation * plot_step_size_validation, 0].reshape(log1.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    # plotting...
    # plt.xlabel('Iteration(stride-{})'.format(plot_step_size_validation), {'size': x_label_scale})
    plt.figure(figsize=(8, 5.5))
    plt.xlabel('Every {} batches.'.format(r'${}$'.format(plot_step_size_validation*10)), {'size': x_label_scale})
    plt.ylabel('Gap to CP-SAT', {'size': y_label_scale})
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
    plt.ylabel('Gap to CP-SAT', {'size': y_label_scale})
    plt.grid()
    x1 = np.array([i + 1 for i in range(obj_incumbent1.shape[0])])
    plt.plot(x1, obj_last_step1, color='tab:blue', label='{}×{}'.format(10, 10))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('merged_last-step_validation_log', save_file_type))
    if show:
        plt.show()


