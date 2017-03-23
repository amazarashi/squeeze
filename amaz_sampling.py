import six
import numpy as np

class Sampling(object):


    def __init__(self):
        pass

    def random_sampling(self,data_length,batch_size,epoch):
        """
        yield indices result of random sampling
        """
        for i in six.moves.range(epoch):
            yield  np.random.permutation(data_length)[:batch_size]

    def random_sampling_label_normarize(self,data_length,batch_size,category_num):
        """
        ### FIX ME ###
        yield indices result of random sampling but the sampled-item
        number is equal dependigng on category
        """
