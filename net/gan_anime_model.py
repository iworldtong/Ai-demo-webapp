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
    
    data_path = './data/'
    num_workers = 4    
    lrg = 2e-4  
    lrd = 2e-4  
    beta1=0.5 
    gpu=True

    d_every = 1 
    g_every = 5 
    save_every = 10 #

    gen_search_num = 1024 # total generate number
    gen_num = 64 # pick number
    gen_mean = 0
    gen_std = 1

opt = Config()
def train(**kwargs):
    for k_,v_ in kwargs.items():
        setattr(opt,k_,v_)
    if opt.vis:
        from visualize import Visualizer
        vis = Visualizer(opt.env)
    
    transforms = tv.transforms.Compose([
                    tv.transforms.Resize(opt.image_size),
                    tv.transforms.CenterCrop(opt.image_size),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
    
    dataset = tv.datasets.ImageFolder(opt.data_path,transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size = opt.batch_size,
                                         shuffle = True,
                                         num_workers= opt.num_workers,
                                         drop_last=True
                                         )

    # nets
    netg, netd = NetG(opt), NetD(opt)  
    map_location=lambda storage, loc: storage
    if opt.netd_path:
        netd.load_state_dict(torch.load(opt.netd_path, map_location = map_location)) 
    if opt.netg_path:
        netg.load_state_dict(torch.load(opt.netg_path, map_location = map_location))

    # optim, losses
    optimizer_g = torch.optim.Adam(netg.parameters(),opt.lrg,betas=(opt.beta1, 0.999))
    optimizer_d = torch.optim.Adam(netd.parameters(),opt.lrd,betas=(opt.beta1, 0.999))
    criterion = torch.nn.BCELoss()

    # noises -- input of generator
    true_labels = Variable(torch.ones(opt.batch_size))
    fake_labels = Variable(torch.zeros(opt.batch_size))
    fix_noises = Variable(torch.randn(opt.batch_size,opt.nz,1,1))
    noises = Variable(torch.randn(opt.batch_size,opt.nz,1,1))

    errord_meter = AverageValueMeter()
    errorg_meter = AverageValueMeter()

    if torch.cuda.is_available() and opt.gpu:
        netd.cuda()
        netg.cuda()
        criterion.cuda()
        true_labels,fake_labels = true_labels.cuda(), fake_labels.cuda()
        fix_noises,noises = fix_noises.cuda(),noises.cuda()
        
    epochs = range(opt.max_epoch)
    for epoch in iter(epochs):
        for ii,(img,_) in tqdm.tqdm(enumerate(dataloader)):
            real_img = Variable(img)
            if torch.cuda.is_available() and opt.gpu: 
                real_img=real_img.cuda()
            if ii%opt.d_every==0:
                # train discriminator
                optimizer_d.zero_grad()
                ## 尽可能的把真图片判别为正确
                output = netd(real_img)
                error_d_real = criterion(output,true_labels)
                error_d_real.backward()
                
                ## 尽可能把假图片判别为错误
                noises.data.copy_(torch.randn(opt.batch_size,opt.nz,1,1))
                fake_img = netg(noises).detach() # get fake image according to the noisy
                output = netd(fake_img)
                error_d_fake = criterion(output,fake_labels)
                error_d_fake.backward()
                optimizer_d.step()

                error_d = error_d_fake + error_d_real

                errord_meter.add(error_d.data[0])

            if ii%opt.g_every==0:
                # train generator
                optimizer_g.zero_grad()
                noises.data.copy_(torch.randn(opt.batch_size,opt.nz,1,1))
                fake_img = netg(noises)
                output = netd(fake_img)
                error_g = criterion(output,true_labels)
                error_g.backward()
                optimizer_g.step()                
                errorg_meter.add(error_g.data[0])

        if epoch%opt.save_every==0:
            torch.save(netd.state_dict(),'checkpoints/netd_%s.pth' %epoch)
            torch.save(netg.state_dict(),'checkpoints/netg_%s.pth' %epoch)
            errord_meter.reset()
            errorg_meter.reset()
            optimizer_g = torch.optim.Adam(netg.parameters(),opt.lrg,betas=(opt.beta1, 0.999))
            optimizer_d = torch.optim.Adam(netd.parameters(),opt.lrd,betas=(opt.beta1, 0.999))

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

