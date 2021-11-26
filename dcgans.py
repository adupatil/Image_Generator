from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# Setting batch size
batchsize = 64
imagesize = 64

#Creating Transformations
transform = transforms.Compose([transforms.Scale(imagesize),
transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    
# Loading dataset
data = dset.CIFAR10(root='./data',download=True,transform=transform)
dataloader = torch.utils.data.DataLoader(data,batch_size=batchsize,shuffle=True)

#Defining weights and functions that are fed to the nn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#Defining generator
class G(nn.Module):
    def __init__(self):
        super(G,self).__init__() # inheriting the nn Module
        # Creating the meta function for defining the neural net
        self.main=nn.Sequential(
            nn.ConvTranspose2d(in_channels=100,out_channels=512,kernel_size=4,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=4,stride=2,padding=1,bias=False),
            nn.Tanh()
            )
            
    def forward(self, input):
        output = self.main(input)
        return output
# Creating instance of generator
netG = G()
netG.apply(weights_init)

# Defining Discriminator
class D(nn.Module):
    def __init__(self):
        super(D,self).__init__()
        self.main=nn.Sequential(
            nn.Conv2d(3,64,4,2,1,bias=True),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64,128,4,2,1,bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(128,256,4,2,1,bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(256,512,4,2,1,bias=True),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(512,1,4,1,0,bias=True),
            nn.Sigmoid()
            )
    def forward(self,input):
        output = self.main(input)
        return output.view(-1)
netD = D()
netD.apply(weights_init)

# Training the models
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizerG = optim.Adam(netG.parameters(),lr=0.0002,betas=(0.5,0.999))

if __name__ == '__main__':
    for epochs in range(25):
        for i,data in enumerate(dataloader,0):
            
            # Step 1 Updating weights for training of Discriminator
            netD.zero_grad()
            
            # Training the discriminator on real images
            real,_ = data
            input = Variable(real)
            target = Variable(torch.ones(input.size()[0]))
            output = netD(input)
            err_real = criterion(output,target)
            
            # Training discriminator on fake images
            noise = Variable(torch.randn(input.size()[0],100,1,1))
            fake = netG(noise)
            target = Variable(torch.zeros(input.size()[0]))
            output = netD(fake.detach())
            err_fake = criterion(output, target)
            
            #Backpropogating the error
            errD = err_real  + err_fake
            errD.backward()
            optimizerD.step()
            
            # Step 2 updating weights for Generator
            netG.zero_grad()
            # training the Generator
            target = Variable(torch.ones(input.size()[0]))
            output = netD(fake)
            errG = criterion(output,target)
            errG.backward()
            optimizerG.step()
            
            #printing results
            print(f"[{epochs}/25][{i}/{len(dataloader)}] LossD:{errD.data[0]} LossG:{errG.data[0]}")
            if i % 100 == 0:
                vutils.save_image(real,"%s/real_sample.png" % './results')
                fake = netG(noise)
                vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epochs))