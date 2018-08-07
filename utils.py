import pathlib
import numpy as np
import chainer
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import chainer.backends.cuda
from chainer import Variable


def combine_images(generated_images):
    total = generated_images.shape[0]
    cols = int(np.sqrt(total))
    rows = int(np.ceil(float(total) / cols))
    width, height = generated_images.shape[1:3]
    combined_image = np.zeros(
        (height * rows, width * cols, 3), dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index / cols)
        j = index % cols
        for ch in range(3):
            combined_image[width*i:width*(i+1), height*j:height*(j+1), ch] =\
                image[:, :, ch]
    return combined_image


def out_generated_image(gen, rows, cols, seed, dst):
    """
    Trainer extension that save Generated data

    Parameters
    -------------
    gen: Model
        Generator

    seed: int
        fix random by value

    dst: PosixPath
        file path to save plotted result

    datasize: int
        the number of plotted datas

    Return
    ------------
    make_image:function
        function that returns make_images that has Trainer object
        as argument.
    """
    @chainer.training.make_extension(trigger=(1, 'epoch'))
    def make_image(trainer):
        n_images = rows * cols

        xp = gen.xp  # get module
        np.random.seed(seed)  # fix seed
        z = Variable(xp.asarray(gen.make_hidden(n_images)))
        with chainer.using_config('train', False):
            with chainer.using_config('enable_backprop', False):
                x = gen(z)
        x = chainer.backends.cuda.to_cpu(x.data)
        np.random.seed()

        x = (x * 127.5 + 127.5) / 255  # 0~255に戻し0~1へ変形
        x = x.transpose(0, 2, 3, 1)  # NCHW->NHWCに変形
        x = combine_images(x)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
        axes.imshow(x)
        axes.axis("off")
        preview_dir = pathlib.Path('{}/preview'.format(dst))
        preview_path = preview_dir /\
            'image_{:}epoch.jpg'.format(trainer.updater.epoch)
        if not preview_dir.exists():
            preview_dir.mkdir()
        axes.set_title("epoch: {}".format(trainer.updater.epoch), fontsize=18)
        fig.tight_layout()
        fig.savefig(preview_path)
        plt.close(fig)

    return make_image
