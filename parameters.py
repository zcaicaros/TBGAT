import argparse

parser = argparse.ArgumentParser(description='Neural-Tabu')

## env parameters
parser.add_argument('--tabu_size', type=int, default=-1)  # -1 = dynamic tabu size, refer to section 3.4 of paper:
# https://www.sciencedirect.com/science/article/pii/S0305054805003989?casa_token=_PxtWhxsw4UAAAAA:IFo4CA7ZBLuTzoTbhMWAOPihkz99jh-Jy3Y9uO3fTLGZVCYUC28Ay8oo2trUvqt7qcp08kyz8IA
parser.add_argument('--mask_previous_action', type=str, default='False', choices=('True', 'False'))
parser.add_argument('--path_finder', type=str, default='pytorch')
parser.add_argument('--init_type', type=str, default='fdd-divide-wkr')  # fdd-divide-wkr, spt
parser.add_argument('--gamma', type=float, default=1)
## TBGAT parameters
parser.add_argument('--in_channels_fwd', type=int, default=3)
parser.add_argument('--in_channels_bwd', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--out_channels', type=int, default=128)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--dropout_for_gat', type=float, default=0)
## which embed net to use, if TPMCAM, its config is the same as that in the paper, the above params are no use.
parser.add_argument('--embed_net', type=str, default='TPMCAM', choices=('TBGAT', 'TPMCAM'))
## training parameters
parser.add_argument('--j', type=int, default=10)
parser.add_argument('--m', type=int, default=10)
parser.add_argument('--l', type=int, default=1)
parser.add_argument('--h', type=int, default=99)
parser.add_argument('--lr', type=float, default=5e-5)  # TPMCAM: 5e-5; TBGAT: 1e-5
parser.add_argument('--steps_learn', type=int, default=10)
parser.add_argument('--transit', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--total_instances', type=int, default=128000)
parser.add_argument('--step_validation', type=int, default=10)
parser.add_argument('--ent_coeff', type=float, default=1e-5)
parser.add_argument('--validation_inst_number', type=int, default=100)
parser.add_argument('--training_seed', type=int, default=6)  # 6
parser.add_argument('--embed_tabu_label', type=str, default='False', choices=('True', 'False'))
# ts_outer: DRL select only from non-tabu (if all tabu, then DRL consider all mv and select)
# ts_inner: if exists non-tabu, DRL select from non-tabu, elif exists mvs meet aspiration criteria, DRL select from
# aspiration, else random select.
parser.add_argument('--action_selection_type', type=str, default='ls', choices=('ls', 'ts_outer', 'ts_inner'))
## testing parameters
parser.add_argument('--test_specific_size', type=str, default='True', choices=('True', 'False'))
parser.add_argument('--test_synthetic', type=str, default='True', choices=('True', 'False'))
parser.add_argument('--t_j', type=int, default=200)
parser.add_argument('--t_m', type=int, default=40)
parser.add_argument('--t_seed', type=int, default=1)
parser.add_argument('--drl_with_tabu', type=str, default='False', choices=('True', 'False'))

args = parser.parse_args()
