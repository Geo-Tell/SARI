class Config(object):
    def __init__(self):
        ## model and loss
        self.ignore_label = 255
        self.aspp_global_feature = False
        ## dataset
        self.n_classes = 5
        self.n_workers = 8
        self.crop_size = (512, 512)
        ## optimizer
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.lr_power = 0.9
        ## training control
        self.scales = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
        self.ims_per_gpu = 4
        self.msg_iter = 100
        self.ohem_thresh = 0.7
        ## eval control
        self.eval_batchsize = 1
        self.eval_n_workers = 2
        self.eval_scales = (0.75, 1.0, 1.25)