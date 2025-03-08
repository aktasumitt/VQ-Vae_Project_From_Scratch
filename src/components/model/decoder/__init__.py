import torch.nn as nn
from src.exception.exception import ExceptionNetwork, sys

class Decoder(nn.Module):
    def __init__(self, channel_size,hidden_dim=256):
        super(Decoder,self).__init__()
        self.hidden_dim=hidden_dim
        self.input_layer=nn.Conv2d(1,hidden_dim,kernel_size=1)

        self.residual_block1=self.residual()
        self.residual_block2=self.residual()
        
        self.convtranspose_block=nn.Sequential(nn.ConvTranspose2d(hidden_dim,hidden_dim,kernel_size=4,stride=2,padding=1,bias=False),
                                               nn.ConvTranspose2d(hidden_dim,channel_size,kernel_size=4,stride=2,padding=1,bias=False))
    
    def residual(self):
        residual=nn.Sequential(nn.ReLU(),
                               nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=3,padding=1,bias=False),
                               nn.ReLU(),
                               nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=1,bias=False))        
        return residual

    def forward(self,x):
        try:
            x=self.input_layer(x)
            r1=self.residual_block1(x)+x
            r2=self.residual_block2(r1)+r1
            out=self.convtranspose_block(r2)
            
            return out
        
        except Exception as e:
            raise ExceptionNetwork(e,sys)
