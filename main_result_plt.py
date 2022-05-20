import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


PBGAT_result = pd.read_excel('./excel/20_128-128-4-0_1e-05-10-500-64-64000-10-1e-05-True.xlsx')
PBGAT_result = np.asarray(PBGAT_result)[:, :2]
PBGAT_result_500 = PBGAT_result[PBGAT_result[:, 1] != -1, :]
PBGAT_upper_half_result_500 = PBGAT_result_500[0:13*4:4, :]

L2S_results_500 = np.concatenate(
    [PBGAT_upper_half_result_500[:, [0]],
     np.array(
         [0.09286241593198157,  # Tai-15x15
          0.11630445289887584,  # Tai-20x15
          0.12011285987528157,  # Tai-20x20
          0.14932057245981312,  # Tai-30x15
          0.1749140315382784,  # Tai-30x20
          0.10193235892475641,  # Tai-50x15
          0.12450978136369906,  # Tai-50x20
          0.07287236966416857,  # Tai-100x20
          0.027812199762473983,  # ABZ-10x10
          0.11939940279867234,  # ABZ-20x15
          0.0,  # FT-6x6
          0.0989247311827957,  # FT-10x10
          0.11931330472103004,  # FT-20x5
          ]
     ).reshape(-1, 1)],
    axis=1
)


RLGNN_result = np.concatenate(
    [PBGAT_upper_half_result_500[:, [0]],
     np.array(
         [0.20133442730138218,  # Tai-15x15
          0.24904817123515918,  # Tai-20x15
          0.2923536490847057,  # Tai-20x20
          0.24695511117046384,  # Tai-30x15
          0.31971817921406853,  # Tai-30x20
          0.15923479092613355,  # Tai-50x15
          0.21295767313883074,  # Tai-50x20
          0.09243435198182624,  # Tai-100x20
          0.10123944925588357,  # ABZ-10x10
          0.2901802608002126,  # ABZ-20x15
          0.2909090909090909,  # FT-6x6
          0.22795698924731184,  # FT-10x10
          0.14849785407725322,  # FT-20x5
          ]
     ).reshape(-1, 1)],
    axis=1
)


ScheduleNet_result = np.concatenate(
    [PBGAT_upper_half_result_500[:, [0]],
     np.array(
         [0.15298000486235536,  # Tai-15x15
          0.19387122165343418,  # Tai-20x15
          0.1723162487838416,  # Tai-20x20
          0.190780150243806,  # Tai-30x15
          0.2372073462964041,  # Tai-30x20
          0.13864820923013993,  # Tai-50x15
          0.13525245504521136,  # Tai-50x20
          0.06656635078557588,  # Tai-100x20
          0.061477473699407564,  # ABZ-10x10
          0.20546332420373906,  # ABZ-20x15
          0.07272727272727272,  # FT-6x6
          0.19462365591397848,  # FT-10x10
          0.28583690987124466,  # FT-20x5
          ]
     ).reshape(-1, 1)],
    axis=1
)

L2D_result = np.concatenate(
    [PBGAT_upper_half_result_500[:, [0]],
     np.array(
         [0.2596457890303421,  # Tai-15x15
          0.29983189110764086,  # Tai-20x15
          0.3159035496412705,  # Tai-20x20
          0.32996651678867445,  # Tai-30x15
          0.335900786133547,  # Tai-30x20
          0.2085921808611566,  # Tai-50x15
          0.23142739891671377,  # Tai-50x20
          0.13518418673730992,  # Tai-100x20
          0.18378489630150335,  # ABZ-10x10
          0.287662364505407,  # ABZ-20x15
          0.2,  # FT-6x6
          0.36666666666666664,  # FT-10x10
          0.38540772532188844,  # FT-20x5
          ]
     ).reshape(-1, 1)],
    axis=1
)

CPSAT_result = np.concatenate(
    [PBGAT_upper_half_result_500[:, [0]],
     np.array(
         [8.077544426494346e-05,  # Tai-15x15
          0.0021135191933276845,  # Tai-20x15
          0.0070089728609427065,  # Tai-20x20
          0.020574668406370667,  # Tai-30x15
          0.028128276715201135,  # Tai-30x20
          0.0,  # Tai-50x15
          0.028331751830319905,  # Tai-50x20
          0.0394992068128869,  # Tai-100x20
          0.0,  # ABZ-10x10
          0.010456304351785286,  # ABZ-20x15
          0.0,  # FT-6x6
          0.0,  # FT-10x10
          0.0,  # FT-20x5
          ]
     ).reshape(-1, 1)],
    axis=1
)

# calculate improvement
against_L2S_500 = (PBGAT_upper_half_result_500[:, 1] - L2S_results_500[:, 1]) / (L2S_results_500[:, 1] + np.finfo(np.float32).eps.item())
against_RLGNN = (RLGNN_result[:, 1] - L2S_results_500[:, 1]) / (RLGNN_result[:, 1] + np.finfo(np.float32).eps.item())
against_ScheduleNet = (ScheduleNet_result[:, 1] - L2S_results_500[:, 1]) / (ScheduleNet_result[:, 1] + np.finfo(np.float32).eps.item())
against_L2D = (L2D_result[:, 1] - L2S_results_500[:, 1]) / (L2D_result[:, 1] + np.finfo(np.float32).eps.item())
# print(against_L2S_500)


x_labels = [
 'Tai 15x15',
 'Tai 20x15',
 'Tai 20x20',
 'Tai 30x15',
 'Tai 30x20',
 'Tai 50x15',
 'Tai 50x20',
 'Tai 100x20',
 'ABZ 10x10',
 'ABZ 20x15',
 'FT 6x6',
 'FT 10x10',
 'FT 20x5'
]

# plot parameters
s, t = 3, 6
x_label_scale = 15
y_label_scale = 20
anchor_text_size = 10
show = True
save = True
save_file_type = '.pdf'
fig, ax = plt.subplots(figsize=(6, 3))
x = np.arange(len(x_labels[s:t])) * 100  # the label locations
width = 20  # the width of the bars

# plotting...

# rects1 = ax.bar(x - width, PBGAT_upper_half_result_500[:8, 1]*100, width, label='PBGAT', color='#f19b61')
# rects2 = ax.bar(x + 0*width, L2S_results_500[:8, 1]*100, width, label='L2S', color='#b0c4de')
# rects3 = ax.bar(x + 1*width, ScheduleNet_result[:8, 1]*100, width, label='ScheduleNet', color='#8fbc8f')
# rects4 = ax.bar(x + 2*width, RLGNN_result[:8, 1]*100, width, label='RLGNN', color='#d18c8d')
# rects5 = ax.bar(x + 3*width, CPSAT_result[:8, 1]*100, width, label='CPSAT', color='#4c72b0')

rects1 = ax.bar(x - width, against_L2S_500[s:t]*100, width, label='Imp to L2S', color='#f19b61')
rects2 = ax.bar(x + 0*width, against_RLGNN[s:t]*100, width, label='Imp to RLGNN', color='#b0c4de')
rects3 = ax.bar(x + 1*width, against_ScheduleNet[s:t]*100, width, label='Imp to ScheduleNet', color='#8fbc8f')
rects4 = ax.bar(x + 2*width, against_L2D[s:t]*100, width, label='Imp to L2D', color='#4c72b0')

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Improvement %', {'size': y_label_scale})
plt.grid(axis='y', zorder=0)
ax.set_xticks(x + width/2)
ax.set_xticklabels(x_labels[s:t], fontsize=15)
ax.yaxis.set_major_formatter(PercentFormatter())
# lg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, shadow=False, ncol=5)
ax.set_axisbelow(True)

padding = 3
fmt = '%.1f'
fontsize = 10
# ax.bar_label(rects1, padding=padding, fmt=fmt, fontsize=fontsize)
# ax.bar_label(rects2, padding=padding, fmt=fmt, fontsize=fontsize)
# ax.bar_label(rects3, padding=padding, fmt=fmt, fontsize=fontsize)
# ax.bar_label(rects4, padding=padding, fmt=fmt, fontsize=fontsize)
# ax.bar_label(rects5, padding=padding, fmt=fmt, fontsize=fontsize)
# ax.text(s="{}%".format(10), ha='center')

ax.bar_label(rects1, padding=padding, fmt=fmt, fontsize=fontsize)
ax.bar_label(rects2, padding=padding, fmt=fmt, fontsize=fontsize)
ax.bar_label(rects3, padding=padding, fmt=fmt, fontsize=fontsize)
ax.bar_label(rects4, padding=padding, fmt=fmt, fontsize=fontsize)

fig.tight_layout()

if save:
    plt.savefig('./{}{}'.format(str(x_labels[s:t]), save_file_type), bbox_inches='tight')
if show:
    plt.show()


# PBGAT_lower_half_result_500 = PBGAT_result_500[13*4::4, :]
# print(PBGAT_lower_half_result_500)
