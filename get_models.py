import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from os.path import join, isfile
import torchvision.models as models
import torch

import seqMatchNet

class WXpB(nn.Module):
    def __init__(self, inDims, outDims):
        super().__init__()
        self.inDims = inDims
        self.outDims = outDims
        self.conv = nn.Conv1d(inDims, outDims, kernel_size=1)

    def forward(self, x):
        x = x.squeeze(-1) # convert [B,C,1,1] to [B,C,1]
        feat_transformed = self.conv(x)
        return feat_transformed.permute(0,2,1) # return [B,1,C]

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

def get_pooling(opt,encoder_dim):

    if opt.pooling:
        global_pool = nn.AdaptiveMaxPool2d((1,1)) # no effect
        poolLayers =  nn.Sequential(*[global_pool, WXpB(encoder_dim, opt.outDims), L2Norm(dim=-1)])
    else:
        global_pool = nn.AdaptiveMaxPool2d((1,1)) # no effect
        poolLayers =  nn.Sequential(*[global_pool, Flatten(), L2Norm(dim=-1)])
    return poolLayers

def get_matcher(opt,device):

    if opt.matcher == 'seqMatchNet':
        sm = seqMatchNet.seqMatchNet()
        matcherLayers = nn.Sequential(*[sm])
    else:
        matcherLayers = None

    return matcherLayers

def printModelParams(model):

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    return
    
def get_model(opt,input_dim,device):
    model = nn.Module()
    encoder_dim = input_dim

    poolLayers = get_pooling(opt,encoder_dim)
    model.add_module('pool', poolLayers)

    matcherLayers = get_matcher(opt,device)
    if matcherLayers is not None:
        model.add_module('matcher',matcherLayers)

    isParallel = False
    if opt.nGPU > 1 and torch.cuda.device_count() > 1:
        model.pool = nn.DataParallel(model.pool)
        isParallel = True

    if not opt.resume:
        model = model.to(device)

    scheduler, optimizer, criterion = None, None, None
    if opt.mode.lower() == 'train':
        if opt.optim.upper() == 'ADAM':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr)#, betas=(0,0.9))
        elif opt.optim.upper() == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr,
                momentum=opt.momentum,
                weight_decay=opt.weightDecay)

            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
        else:
            raise ValueError('Unknown optimizer: ' + opt.optim)

        # used only when matcher is none
        criterion = nn.TripletMarginLoss(margin=opt.margin**0.5, p=2, reduction='sum').to(device)

    if opt.resume:
        if opt.ckpt.lower() == 'latest':
            resume_ckpt = join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')
        elif opt.ckpt.lower() == 'best':
            resume_ckpt = join(opt.resume, 'checkpoints', 'model_best.pth.tar')

        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            opt.update({"start_epoch" : checkpoint['epoch']}, allow_val_change=True)
            best_metric = checkpoint['best_score']
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)
            if opt.mode == 'train':
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))

    return model, optimizer, scheduler, criterion, isParallel, encoder_dim
