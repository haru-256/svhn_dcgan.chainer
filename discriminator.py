import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L


class Discriminator(chainer.Chain):
    """Discriminator

    build Discriminator model

    Parametors
    ---------------------
    in_ch: int
        Channel of input image

    bottom_width: int
        width & height of input image

    wscale: float
        std of normal initializer
    Attributes
    ---------------------
    """

    def __init__(self, in_ch=3, bottom_width=4, wscale=0.02):
        super(Discriminator, self).__init__()
        self.bottom_width = bottom_width
        self.in_ch = in_ch
        with self.init_scope():
            # initializers
            w = chainer.initializers.HeNormal(wscale)

            # register layer with variable
            self.c0 = L.Convolution2D(
                in_channels=self.in_ch,
                out_channels=128,
                ksize=5,
                stride=2,
                pad=2,
                initialW=w)  # (N, 128, 16, 16)
            self.c1 = L.Convolution2D(
                in_channels=None,
                out_channels=256,
                ksize=5,
                stride=2,
                pad=2,
                initialW=w,
                nobias=True)  # (N, 256, 8, 8)
            self.c2 = L.Convolution2D(
                in_channels=None,
                out_channels=512,
                ksize=5,
                stride=2,
                pad=2,
                initialW=w,
                nobias=True)  # (N, 512, 4, 4)
            self.l3 = L.Linear(in_size=None, out_size=1, initialW=w)
            self.bn1 = L.BatchNormalization(size=256)
            self.bn2 = L.BatchNormalization(size=512)

    def __call__(self, x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        logits = self.l3(h)

        return logits


if __name__ == "__main__":
    import chainer.computational_graph as c
    from chainer import Variable
    import numpy as np

    z = np.random.uniform(-1, 1, (10, 3, 32, 32)).astype("f")
    model = Discriminator()
    img = model(Variable(z))
    # print(img)
    g = c.build_computational_graph(img)
    with open('dis_graph.dot', 'w') as o:
        o.write(g.dump())
