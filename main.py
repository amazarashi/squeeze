import argparse

import squeeze_net
import amaz_trainer
import amaz_cifar10_dl

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='cifar10')
    parser.add_argument('--epoch', '-e', type=int,
                        default=300,
                        help='maximum epoch')
    parser.add_argument('--batch', '-b', type=int,
                        default=64,
                        help='mini batch number')
    parser.add_argument('--gpu', '-g', type=int,
                        default=-1,
                        help='-1 means cpu, put gpu id here')
    parser.add_argument('--lr', '-lr', type=float,
                        default=0.1,
                        help='leraning rate')

    args = parser.parse_args().__dict__
    lr = args.pop('lr')

    model = squeeze_net.SqueezeNet(10)
    optimizer = optimizer.Adam()
    dataset = amaz_cifar10_dl.Cifar10().loader()
    args['model'] = model
    args['optimizer'] = optimizer
    args['dataset'] = dataset
    main = amaz_trainer.Trainer(**args)
    main.run()