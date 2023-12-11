
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_loss', default=0.001, type=float,)

    parser.add_argument('--dataset', default='CGMNIST', type=str,
                        help='CGMNIST')
    parser.add_argument('--modulation', default='Normal', type=str,
                        choices=['Normal'])

    parser.add_argument('--fps', default=1, type=int, help='Extract how many frames in a second')
    parser.add_argument('--num_frame', default=3, type=int, help='use how many frames for train')

    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--optimizer', default='SGD', type=str)

    parser.add_argument('--learning_rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--ckpt_path', default=None, type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', action='store_true', help='whether to visualize')
    parser.add_argument('--logs_path', default='logs', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)

    parser.add_argument('--gpu', type=int, default=0)  # gpu
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--regbn', action='store_true', help='turn on regbn mode')


    # args = parser.parse_args()
    #
    # args.use_cuda = torch.cuda.is_available() and not args.no_cuda

    return parser.parse_args()

args = get_arguments()

