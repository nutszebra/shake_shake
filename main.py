import sys
sys.path.append('./trainer')
import argparse
import nutszebra_cifar10
import shake_shake
import nutszebra_data_augmentation
import nutszebra_optimizer

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='cifar10')
    parser.add_argument('--load_model', '-m',
                        default=None,
                        help='trained model')
    parser.add_argument('--load_optimizer', '-o',
                        default=None,
                        help='optimizer for trained model')
    parser.add_argument('--load_log', '-l',
                        default=None,
                        help='optimizer for trained model')
    parser.add_argument('--save_path', '-p',
                        default='./',
                        help='model and optimizer will be saved every epoch')
    parser.add_argument('--epoch', '-e', type=int,
                        default=300,
                        help='maximum epoch')
    parser.add_argument('--batch', '-b', type=int,
                        default=128,
                        help='mini batch number')
    parser.add_argument('--gpu', '-g', type=int,
                        default=-1,
                        help='-1 means cpu mode, put gpu id here')
    parser.add_argument('--start_epoch', '-s', type=int,
                        default=1,
                        help='start from this epoch')
    parser.add_argument('--train_batch_divide', '-trb', type=int,
                        default=2,
                        help='divid batch number by this')
    parser.add_argument('--test_batch_divide', '-teb', type=int,
                        default=2,
                        help='divid batch number by this')
    parser.add_argument('--lr', '-lr', type=float,
                        default=0.2,
                        help='leraning rate')
    parser.add_argument('--dim', '-dim', type=int,
                        default=64,
                        help='width')

    args = parser.parse_args().__dict__
    lr = args.pop('lr')
    dim = args.pop('dim')

    print('generating model')
    model = shake_shake.ShakeShake(10, (dim, dim * 2, dim * 4), (4, 4, 4))
    print('Done')
    print('parameters: {}'.format(model.count_parameters()))
    optimizer = nutszebra_optimizer.OptimizerPyramidalResNet(model, lr=lr)
    args['model'] = model
    args['optimizer'] = optimizer
    args['da'] = nutszebra_data_augmentation.DataAugmentationCifar10NormalizeSmall
    main = nutszebra_cifar10.TrainCifar10(**args)
    main.run()
