#coding : utf-8
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import time

import amaz_sampling
import amaz_util
import amaz_sampling
import amaz_datashaping

sampling = amaz_sampling.Sampling()

class Trainer(object):

    def __init__(self,model=None,optimizer=None,dataset=None,epoch=300,batch=128,gpu=-1):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.epoch = epoch
        self.batch = batch
        self.train_x,self.train_y,self.test_x,self.test_y,self.meta = self.init_dataset()
        self.gpu = self.gpu
        self.xp = self.check_cupy(self.gpu)
        self.utility = amaz_util.Utility()
        self.datashaping = amaz_datashaping.DataShaping(self.xp)

    def check_cupy(self,gpu):
        if gpu == -1:
            return np
        else:
            cuda.get_device(gpu).use()
            self.model.to_gpu()
            return cuda.cupy

    def init_dataset(self):
        train_x = self.dataset["train_x"]
        train_y = self.dataset["train_y"]
        test_x = self.dataset["test_x"]
        test_y = self.dataset["test_y"]
        meta = self.dataset["meta"]
        return (train_x,train_y,test_x,test_y,meta)

    def train_one(self):
        model = self.model
        optimizer = self.optimizer
        batch = self.batch
        train_x = self.train_x
        train_y = self.train_y
        meta = self.meta

        sum_loss = 0
        progress = self.utility.create_progressbar(int(len(train_x)),desc='train',stride=1)
        train_data_yeilder = sampling(train_x,train_y,batch,len(meta))
        for _,indices in zip(progress,train_data_yeilder):
            model.cleargrads()
            x = train_x[indices]
            t = train_y[indices]

            x = datashaping.input(x,dtype=np.float32)
            t = datashaping.input(t,dtype=np.int32)

            y = model(x,train=True)
            loss = model.calc_loss(y,t)
            loss.backword()
            loss.to_cpu()
            sum_loss += loss.data * len(x)

            del loss,x,t
            optimizer.update()

    def test_one(self):
        model = self.model
        optimizer = self.optimizer
        batch = self.batch
        test_x = self.test_x
        test_y = self.test_y
        meta = self.meta

        sum_loss = 0
        progress = self.utility.create_progressbar(int(len(train_x)),desc='train',stride=batch)
        for i in progress:
            x = test_x[i:i+batch]
            t = test_y[i:i+batch]

            x = datashaping.input(x,dtype=np.float32)
            t = datashaping.input(t,dtype=np.int32)

            y = model(x,train=False)
            loss = model.calc_loss(y,t)
            sum_loss += loss.data * len(x)

            del loss,x,t

    def run(self):
        epoch = self.epoch

        progressor = self.utility.create_progressbar(epoch + 1,desc='epoch',stride=1,start=0)
        for i in progressor:
            self.train_one():
            self.test_one():
