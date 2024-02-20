import argparse

parser = argparse.ArgumentParser(description='PyTorch RE-SORT Training')

parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')


parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--cos', '--cosine_lr', default=1, type=int,
                     help='lr decay by decay', dest='cos')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                     help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=2023, type=int,
                    help='seed for initializing training. ')




# for number of fourier spaces
parser.add_argument ('--num_f', type=int, default=3, help = 'number of fourier spaces')

parser.add_argument ('--sample_rate', type=float, default=1.0, help = 'sample ratio of the features involved in balancing')
parser.add_argument ('--lrbl', type = float, default = 0.8, help = 'learning rate of balance')

parser.add_argument ('--lambdap', type = float, default = 80.0, help = 'weight decay for weight1 ')
parser.add_argument ('--lambdapre', type = float, default = 1, help = 'weight for pre_weight1 ')

parser.add_argument ('--epochb', type = int, default = 6, help = 'number of epochs to balance')

parser.add_argument ('--lrwarmup_epo', type=int, default=0, help = 'the dim of each feature')
parser.add_argument ('--lrwarmup_decay', type=int, default=0.1, help = 'the dim of each feature')

parser.add_argument ('--n_levels', type=int, default=1, help = 'number of global table levels')

parser.add_argument ('--lambda_decay_rate', type=float, default=0.9, help = 'ratio of epoch for lambda to decay')
parser.add_argument ('--lambda_decay_epoch', type=int, default=5, help = 'number of epoch for lambda to decay')
parser.add_argument ('--min_lambda_times', type=float, default=0.001, help = 'number of global table levels')
parser.add_argument ('--cnn_lossb_lambda', type=float, default=0, help = 'lambda for lossb')

parser.add_argument ('--moments_lossb', type=float, default=1, help = 'number of moments')

parser.add_argument ('--first_step_cons', type=float, default=1, help = 'constrain the weight at the first step')

parser.add_argument ('--decay_pow', type=float, default=2, help = 'value of pow for weight decay')

# for second order moment weight
parser.add_argument ('--second_lambda', type=float, default=0.3, help = 'weight lambda for second order moment loss')
parser.add_argument ('--third_lambda', type=float, default=0.05, help = 'weight lambda for second order moment loss')

# for lr decay epochs
parser.add_argument ('--epochs_decay', type=list, default=[24, 30], help = 'weight lambda for second order moment loss')
parser.add_argument ('--gray_scale', type=float, default=0.1, help = 'weight lambda for second order moment loss')

parser.add_argument('--sum', type=bool, default=True, help='sum or concat')
parser.add_argument('--concat', type=int, default=0, help='sum or concat')
parser.add_argument('--min_scale', type=float, default=0.8, help='')
parser.add_argument('--presave_ratio', type=float, default=0.9, help='the ratio for presaving features')

