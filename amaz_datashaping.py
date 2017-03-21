import numpy as np
from chainer import cuda

class DataShaping(object):

    def __init__(self,xp):
        self.xp = xp

    def input(self,data,dtype):
        """
        prepare input data for model
        """
        xp = self.xp

        return
