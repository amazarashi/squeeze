import urllib.request
import tarfile
from os import system
import os
import sys
import six
import pickle
from tqdm import tqdm

class Log(object):

    def __init__(self):
        self.name = "log"
        self.model_save_path = "./model"
        self.log_save_path = "./log"
        self.train_loss_path = "log/train_loss.txt"
        self.test_loss_path = "log/test_loss.txt"
        self.accuracy_path = "log/accuracy.txt"
        self.init = self.init_log_env()
        self.train_loss_fp = open(self.train_loss_path, "w")
        self.test_loss_fp = open(self.test_loss_path, "w")
        self.accuracy_fp = open(self.accuracy_path, "w")
        self.init_log_file = self.init_log_file()


    def init_log_env(self):
        if os.path.exists(self.model_save_path) == False:
            os.mkdir("model")
        if os.path.exists(self.log_save_path) == False:
            os.mkdir("log")
        return

    def init_log_file(self):
        self.train_loss_fp.write("epoch\ttrain_loss\n")
        self.test_loss_fp.write("epoch\ttest_loss\n")
        self.accuracy_fp.write("epoch\taccuracy\n")
        return

    def finish_log(self):
        self.train_loss.close()
        self.test_loss.close()
        self.accuracy.close()
        return

    def save_model(self,model,epoch):
        pickle.dump(model, open(self.model_save_path+"/model_{0}.pkl".format(str(epoch)), "wb"), -1)
        return

    def train_loss(self,epoch,loss):
        self.train_loss_fp.write("%d\t%f\n" % (epoch, loss))
        self.train_loss_fp.flush()
        return

    def test_loss(self,epoch,loss):
        self.test_loss_fp.write("%d\t%f\n" % (epoch, loss))
        self.test_loss_fp.flush()
        return

    def accuracy(self,epoch,accuracy):
        self.accuracy_fp.write("%d\t%f\n" % (epoch, accuracy))
        self.accuracy_fp.flush()
        return

    def plt_loss(self):
        return

    def plt_accuracy(self):
        return
