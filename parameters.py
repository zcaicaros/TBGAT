import argparse

parser = argparse.ArgumentParser(description='Neural-Tabu')

# env parameters
parser.add_argument('--tabu_size', type=int, default=20)
parser.add_argument('--mask_previous_action', type=str, default='False', choices=('True', 'False'))
parser.add_argument('--path_finder', type=str, default='pytorch')
parser.add_argument('--init_type', type=str, default='fdd-divide-wkr')
parser.add_argument('--gamma', type=float, default=1)
# model parameters
parser.add_argument('--in_channels_fwd', type=int, default=3)
parser.add_argument('--in_channels_bwd', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--out_channels', type=int, default=128)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--dropout_for_gat', type=float, default=0)
# training parameters
parser.add_argument('--j', type=int, default=10)
parser.add_argument('--m', type=int, default=10)
parser.add_argument('--l', type=int, default=1)
parser.add_argument('--h', type=int, default=99)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--steps_learn', type=int, default=10)
parser.add_argument('--transit', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--total_instances', type=int, default=32000)
parser.add_argument('--step_validation', type=int, default=10)
parser.add_argument('--ent_coeff', type=float, default=1e-5)
parser.add_argument('--validation_inst_number', type=int, default=100)
parser.add_argument('--training_seed', type=int, default=6)
parser.add_argument('--embed_tabu_label', type=str, default='True', choices=('True', 'False'))
# testing parameters
parser.add_argument('--test_specific_size', type=str, default='True', choices=('True', 'False'))
parser.add_argument('--test_synthetic', type=str, default='False', choices=('True', 'False'))
parser.add_argument('--t_j', type=int, default=10)
parser.add_argument('--t_m', type=int, default=10)
parser.add_argument('--t_seed', type=int, default=1)

args = parser.parse_args()
