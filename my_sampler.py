##https://stackoverflow.com/questions/60993677/how-can-i-save-pytorchs-dataloader-instance

import torch
import random
from torch.utils.data.dataloader import RandomSampler

mseed = 110598

class MySampler(RandomSampler):
    def __init__(self, data, i=0, batch_size=1):
        super(MySampler, self).__init__(data,replacement=False)
        self.start_index = i
        self.batch_size  = batch_size
        self.data = data

    def __iter__(self):
        n = len(self.data)
        self.start_index = self.start_index * self.batch_size % n 
        ## Seed this
        torch.manual_seed(mseed)   
        rnd_index = torch.randperm(n).tolist()
        if self.start_index != 0:
            rnd_index = rnd_index[self.start_index:]
            self.start_index = 0
        return iter(rnd_index)
