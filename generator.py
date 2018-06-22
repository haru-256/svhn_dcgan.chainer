import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L


class Generator(chainer.Chain):
    """Generator

    build Generator model

    Parametors
    ---------------------
    n_hidden: int
       dims of random vector z

    bottom_width: int
       Width when converting the output of the first layer
       to the 4-dimensional tensor

    in_ch: int
       Channel when converting the output of the first layer
       to the 4-dimensional tensor

    Attributes
    ---------------------
    """

    def __init__(self, n_hidden=100, bottom_width=4, in_ch=512, wscale=0.02):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.bottom_width = bottom_width
        self.in_ch = in_ch

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)  # initializers

            self.l0 = L.Linear(in_size=self.n_hidden,
                               out_size=bottom_width*bottom_width*in_ch,
                               initialW=w,
                               nobias=True)  # (N, bottom_width*bottom_width*in_ch)
            self.dc1 = L.Deconvolution2D(in_channels=None,
                                         out_channels=in_ch//2,
                                         pad=1,
                                         stride=2,
                                         ksize=4,
                                         initialW=w,
                                         nobias=True)  # (N, 256, 8, 8)
            self.dc2 = L.Deconvolution2D(in_channels=None,
                                         out_channels=in_ch//4,
                                         pad=1,
                                         stride=2,
                                         ksize=4,
                                         initialW=w,
                                         nobias=True)  # (N, 128, 16, 16)
            self.dc3 = L.Deconvolution2D(in_channels=None,
                                         out_channels=3,
                                         pad=1,
                                         stride=2,
                                         ksize=4,
                                         initialW=w)  # (N, 3, 32, 32)

            self.bn0 = L.BatchNormalization(
                size=bottom_width*bottom_width*in_ch)
            self.bn1 = L.BatchNormalization(size=in_ch//2)
            self.bn2 = L.BatchNormalization(size=in_ch//4)

    def make_hidden(self, batchsize=100):
        """
        Function that makes z random vector in accordance with the uniform(-1, 1)

        batchsize: int
           batchsize indicate len(z)
        """
        return np.random.uniform(-1, 1, (batchsize, self.n_hidden))\
                        .astype(np.float32)

    def __call__(self, z):
        h = F.relu(self.bn0(self.l0(z)))
        h = F.reshape(
            h, (-1, self.in_ch, self.bottom_width, self.bottom_width))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        x = F.tanh(self.dc3(h))  # linear projection to 2

        return x


if __name__ == "__main__":
    import chainer.computational_graph as c
    from chainer import Variable

    model = Generator(n_hidden=100)
    img = model(Variable(model.make_hidden(10)))
    # print(img)
    g = c.build_computational_graph(img)
    with open('gen_graph.dot', 'w') as o:
        o.write(g.dump())
