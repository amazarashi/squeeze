#coding : utf-8
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import time

import amaz_sampling
import amaz_util

class Trainer(object):

    def __init__(self,model=None,optimizer=None,dataset=None,epoch=300,batch=128,gpu=-1):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.epoch = epoch
        self.batch = batch
        self.gpu = gpu
        self.utility = amaz_util.Utility()

    def train_one(self):
        return


    def test_one(self):
        return


    def run(self):
        # GPU setting
        if gpu_flag >= 0:
             cuda.check_cuda_available()
        xp = cuda.cupy if gpu_flag >= 0 else np

        if self.gpu_flag >= 0:
            cuda.get_device(gpu_flag).use()
            self.model.to_gpu()

        epoch = self.epoch

        progressor = self.utility.create_progressbar(epoch + 1,desc='epoch',stride=1,start=0)
        for i in progressor:
            self.train_one():
            self.test_one():
