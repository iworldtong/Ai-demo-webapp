#coding:utf8
import torch
from torch import nn
import torchvision as tv
from torch.autograd import Variable



class Config(object):    
    batch_size = 256
    image_size = 96  # img size
    max_epoch = 400    
    ngf = 64  # g: feature map number
    ndf = 64  # d: feature map number
    nz = 100  # noisy dim
    
    gen_search_num = 1024 # total generate number
    gen_num = 64 # pick number
    gen_mean = 0
    gen_std = 1

opt = Config()

def generate(**kwargs):
    '''
    Randomly generate anime profile, according the netd to choose better images.
    '''
    for k_,v_ in kwargs.items():
        setattr(opt,k_,v_)
    
    netg, netd = NetG(opt).eval(), NetD(opt).eval()  
    noises = torch.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean, opt.gen_std)
    noises = Variable(noises, volatile=True)

    map_location = lambda storage, loc: storage     #!!!!!!
    if opt.netd_path is not None:
        netd.load_state_dict(torch.load(opt.netd_path, map_location = map_location))
    if opt.netg_path is not None:
        netg.load_state_dict(torch.load(opt.netg_path, map_location = map_location))
    
    if torch.cuda.is_available():
        netd.cuda()
        netg.cuda()
        noises = noises.cuda()
        
    # Generate images, calc discriminator's scores
    fake_img = netg(noises)
    scores = netd(fake_img).data

    # Pick better images
    indexs = scores.topk(opt.gen_num)[1]
    result = []
    for i in indexs:
        result.append(fake_img.data[i])
    
    # save images
    tv.utils.save_image(torch.stack(result), opt.save_path, normalize=True, range=(-1,1))

    return 




'''
Define net structure
'''

class NetG(nn.Module):
    '''
    Generator
    '''
    def __init__(self, opt):
        super(NetG, self).__init__()
        ngf = opt.ngf  # feature map number

        self.main = nn.Sequential(
            # input shape: 1 x 1 x nz
            nn.ConvTranspose2d(opt.nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # output shape: (ngf*8) x 4 x 4

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # output shape: (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # output shape: (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # output shape: (ngf) x 32 x 32

            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()  # use tanh to set output in [-1, 1] 
            # output shape: 3 x 96 x 96
        )

    def forward(self, z):
        return self.main(z)


class NetD(nn.Module):
    '''
    Discriminator
    '''
    def __init__(self, opt):
        super(NetD, self).__init__()
        ndf = opt.ndf
        self.main = nn.Sequential(
            # input shape: 3 x 96 x 96
            nn.Conv2d(3, ndf, 5, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # output shape: (ngf) x 32 x 32
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # output shape: (ngf*2) x 16 x 16
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # output shape: (ngf*4) x 8 x 8
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # output shape: (ngf*8) x 4 x 4
            
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # output probability --- an single number
        )

    def forward(self, x):
        return self.main(x).view(-1)

