import argparse

class Options:
    def __init__(self):
        self.options = None
        self.parser = argparse.ArgumentParser()
        
        self.parser.add_argument('--height',
                                 help='height of input images',
                                 type=int,
                                 default=320)#224
        self.parser.add_argument('--width',
                                 help='width of input images',
                                 type=int,
                                 default=608)#480
        
        self.parser.add_argument('--fill_mode',
                                 help='occlusion filling method',
                                 type=str,
                                 choices=['background', 'opencv', 'inpaint'],
                                 default='background')
        
        self.parser.add_argument('--max_disparity',
                                 help='maximum disparity',
                                 type=int,
                                 default=192)

        self.parser.add_argument('--batch_size',
                                 help='number of images in each batch',
                                 type=int,
                                 default=2)

        self.parser.add_argument('--training_steps',
                                 help='number of steps to train for',
                                 type=int,
                                 default=150000) #250000
        
        self.parser.add_argument('--save_freq',
                                 help='sets the frequency of saving weight',
                                 type=int,
                                 default=2500)
        
        self.parser.add_argument('--model_name',
                                 help='default model name',
                                 type=str,
                                 default='untitled')

        self.parser.add_argument('--network_variant',
                                 help='choose the network to use',
                                 type=str,
                                 default='original') # or 'new'

        self.parser.add_argument('--feature_size', 
                                  help='feature output channel size',
                                  type=int,
                                  default=32)
        # self.parser.add_argument('--mode',
        #                     help='training or inference mode',
        #                     type=str,
        #                     choices=['train', 'inference'],
        #                     default='train')

        # self.parser.add_argument('--sampling_size',
        #                          type=float,
        #                          default=1.0)

        self.parser.add_argument('--training_dataset',
                                 help='dataset to train from',
                                 choices = ['mscoco2','mscoco', 'kitti2015', 'middlebury'],
                                 default= 'mscoco')

### Benchmark stuff
        self.parser.add_argument('--error_log',
                                 help='default error log to load',
                                 type=str,
                                 default='default_errors')
        ## If this quick bench, then no log will be saved 
        self.parser.add_argument('--quick_bench',
                                 help='benchmark the latest weight only',
                                 type=int,
                                 default=1)

        ## 
        self.parser.add_argument('--save_error_image',
                                 help='Save the EPE error image (quick_bench needed)',
                                 type=int,
                                 default=0)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
        
            