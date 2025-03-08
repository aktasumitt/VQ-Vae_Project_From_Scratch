import torch.nn as nn
from src.exception.exception import ExceptionNetwork, sys

class Encoder(nn.Module):
    def __init__(self, channel_size, hidden_dim=256):
        super(Encoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(channel_size, hidden_dim, kernel_size=4, stride=2, padding=1,bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False))
        
        self.residual_block1 = self.residual()
        self.residual_block2 = self.residual()

        self.last_layer= nn.Conv2d(hidden_dim,1,kernel_size=1)
    
    def residual(self):
        return nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1,bias=False)
        )

    def forward(self, x):
        try:
            x = self.conv_block(x)
            r1 = self.residual_block1(x) + x
            r2 = self.residual_block2(r1) + r1
            out= self.last_layer(r2)
            
            return out
        except Exception as e:
            raise ExceptionNetwork(e, sys)
