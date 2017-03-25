import chainer
from chainer import optimizers

class Optimizers(object):

    def __init__(self,model,epoch=300):
        self.model = model
        self.epoch = epoch
        self.optimizer = None

    def __call__(self):
        pass

    def update(self):
        self.optimizer.update()

    def setup(self,model):
        self.optimizer.setup(model)

class OptimizerSqueeze(Optimizers):

    def __init__(self,model=None,lr=0.01,momentum=0.9,epoch=300):
        super(OptimizerSqueeze,self).__init__(model,epoch)
        self.lr = lr
        self.optimizer = optimizers.MomentumSGD(self.lr,momentum)

    def update_parameter(self,current_epoch):
        if current_epoch >= int(self.epoch/3):
            new_lr = 0.01
            self.optimizer.lr = new_lr
            print("optimizer was changed to {0}..".format(new_lr))
        elif current_epoch >= int(self.epoch*2/3):
            new_lr = 0.001
            self.optimizer.lr = new_lr
            print("optimizer was changed to {0}..".format(new_lr))
