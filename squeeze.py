import chainer
import chainer.functions as F
import chainer.links as L
import skimage.io as io
import numpy as np
from chainer import utils

class FireModule(chainer.Chain):

    def __init__(self,input_size,s1,e1,e3):
        super(FireModule,self).__init__(
            conv1 = L.Convolution2D(input_size,s1,1),
            conv2 = L.Convolution2D(s1,e1,1),
            conv3 = L.Convolution2D(s1,e3,3,pad=(1,1)),
        )

    def __call__(self,x):
        h = F.relu(self.conv1(x))
        h1 = self.conv2(h)
        h2 = self.conv3(h)
        h_expand = F.concat([h1,h2],axis=1)
        return F.relu(h_expand)

class Squeeze(chainer.Chain):

    def __init__(self,category_num=10):
        super(Squeeze,self).__init__(
            conv1 = L.Convolution2D(3,96,7,stride=2),
            fire2 = FireModule(96,16,64,64),
            fire3 = FireModule(128,16,64,64),
            fire4 = FireModule(128,32,128,128),
            fire5 = FireModule(256,32,128,128),
            fire6 = FireModule(256,48,192,192),
            fire7 = FireModule(384,48,192,192),
            fire8 = FireModule(384,64,256,256),
            fire9 = FireModule(512,64,256,256),
            conv10 = L.Convolution2D(512,category_num,1,stride=1),
        )

    def __call__(self,x):
        #x = chainer.Variable(x)
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h,3,stride=2)

        h = self.fire2(h)
        h = self.fire3(h)
        h = self.fire4(h)
        h = F.max_pooling_2d(h,3,stride=2)

        h = self.fire5(h)
        h = self.fire6(h)
        h = self.fire7(h)
        h = F.max_pooling_2d(h,3,stride=2)

        h = self.fire8(h)
        h = F.max_pooling_2d(h,3,stride=2)

        h = self.fire9(h)
        h = F.dropout(h,ratio=0.5)

        h = self.conv10(h)
        num, categories, y, x = h.data.shape
        h = F.reshape(F.average_pooling_2d(h,(y, x)), (num, categories))

        return h


if __name__ == "__main__":
    imgpath = "/Users/suguru/Desktop/test.jpg"
    img = io.imread(imgpath)
    img = np.asarray(img).transpose(2,0,1).astype(np.float32)/255.
    img = img[np.newaxis]
    ex = model(img)
    print(ex)