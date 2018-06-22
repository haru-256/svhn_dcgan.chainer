import chainer
from chainer import training
from chainer.training import extensions
from generator import Generator
from utils import out_generated_image
import argparse
import pathlib
import matplotlib.pyplot as plt
plt.style.use("ggplot")

if __name__ == '__main__':
    # パーサーを作る
    parser = argparse.ArgumentParser(
        prog='train',  # プログラム名
        usage='train DCGAN',  # プログラムの利用方法
        description='description',  # 引数のヘルプの前に表示
        epilog='end',  # 引数のヘルプの後で表示
        add_help=True,  # -h/–help オプションの追加
    )

    # 引数の追加
    parser.add_argument('-s', '--seed', help='seed',
                        type=int, required=True)
    parser.add_argument('-n', '--number', help='the number of experiments.',
                        type=int, required=True)
    parser.add_argument('--hidden', help='the number of codes of Generator.',
                        type=int, default=100)
    parser.add_argument('-e', '--epoch', help='the number of epoch, defalut value is 100',
                        type=int, default=100)
    parser.add_argument('-bs', '--batch_size', help='batch size. defalut value is 128',
                        type=int, default=128)
    parser.add_argument('-g', '--gpu', help='specify gpu by this number. defalut value is 0',
                        choices=[-1, 0, 1], type=int, default=0)
    parser.add_argument('-dis', '--discriminator',
                        help='specify discriminator by this number. any of following;'
                        ' 0: original, 1: minibatch discriminatio, 2: feature matching, 3: Global Average Pooling. defalut value is 0',
                        choices=[0, 1, 2, 3], type=int, default=0)
    parser.add_argument('-ts', '--tensor_shape',
                        help='specify Tensor shape by this numbers. first args denotes to B, seconds to C.'
                        ' defalut value are B:32, C:8',
                        type=int, default=[32, 8], nargs=2)
    parser.add_argument('-V', '--version', version='%(prog)s 1.0.0',
                        action='version',
                        default=False)

    # 引数を解析する
    args = parser.parse_args()

    gpu = args.gpu
    batch_size = args.batch_size
    n_hidden = args.hidden
    epoch = args.epoch
    seed = args.seed
    number = args.number  # number of experiments
    out = pathlib.Path("result_{0}".format(number))
    if not out.exists():
        out.mkdir()
    out /= pathlib.Path("result_{0}_{1}".format(number, seed))
    if not out.exists():
        out.mkdir()

    # 引数(ハイパーパラメータの設定)の書き出し
    with open(out / "args.txt", "w") as f:
        f.write(str(args))

    print('GPU: {}'.format(gpu))
    print('# Minibatch-size: {}'.format(batch_size))
    print('# n_hidden: {}'.format(n_hidden))
    print('# epoch: {}'.format(epoch))
    print('# out: {}'.format(out))

    # import discrimination
    if args.discriminator == 0:
        print("# Original Discriminator")
        from discriminator import Discriminator
        from updater import DCGANUpdater
        dis = Discriminator()
    elif args.discriminator == 1:
        print("# Discriminator applied Minibatch Discrimination")
        print('# Tensor shape is A x {0} x {1}'.format(
            args.tensor_shape[0], args.tensor_shape[1]))
        from discriminator_md import Discriminator
        from updater import DCGANUpdater
        dis = Discriminator(B=args.tensor_shape[0], C=args.tensor_shape[1])
    elif args.discriminator == 2:
        print("# Discriminator applied matching")
        from discriminator_fm import Discriminator
        from updater_fm import DCGANUpdater
        dis = Discriminator()
    elif args.discriminator == 3:
        print("# Discriminator applied GAP")
        from discriminator_gap import Discriminator
        from updater import DCGANUpdater
        dis = Discriminator()

    print('')

    # Set up a neural network to train
    gen = Generator(n_hidden=n_hidden)
    dis = Discriminator()

    if gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(gpu).use()
        gen.to_gpu()  # Copy the model to the GPU
        dis.to_gpu()

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        # optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer

    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    # Load the GMM dataset
    dataset, _ = chainer.datasets.get_svhn(withlabel=False, scale=255.)
    train_iter = chainer.iterators.SerialIterator(dataset, batch_size)
    print("# Data size: {}".format(len(dataset)), end="\n\n")

    # Set up a trainer
    updater = DCGANUpdater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={
            'gen': opt_gen,
            'dis': opt_dis
        },
        device=gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

    snapshot_interval = (5, 'epoch')
    display_interval = (1, 'epoch')
    trainer.extend(extensions.dump_graph("gen/loss", out_name="gen.dot"))
    trainer.extend(extensions.dump_graph("dis/loss", out_name="dis.dot"))
    trainer.extend(
        extensions.snapshot(
            filename='snapshot_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(
        extensions.snapshot_object(gen, 'gen_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(
        extensions.snapshot_object(dis, 'dis_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PrintReport([
            'epoch',
            'iteration',
            'gen/loss',
            'dis/loss',
        ]),
        trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=100))
    trainer.extend(
        extensions.PlotReport(
            ['gen/loss', 'dis/loss'],
            x_key='epoch',
            file_name='loss_{0}_{1}.jpg'.format(number, seed),
            grid=False))
    trainer.extend(out_generated_image(
        gen, 7, 7, seed, out), trigger=display_interval)

    # Run the training
    trainer.run()
