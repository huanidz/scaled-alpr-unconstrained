import torch
import torch.nn as nn

"""
def res_block(x,sz,filter_sz=3,in_conv_size=1):
	xi  = x
	for i in range(in_conv_size):
		xi  = Conv2D(sz, filter_sz, activation='linear', padding='same')(xi)
		xi  = BatchNormalization()(xi)
		xi 	= Activation('relu')(xi)
	xi  = Conv2D(sz, filter_sz, activation='linear', padding='same')(xi)
	xi  = BatchNormalization()(xi)
	xi 	= Add()([xi,x])
	xi 	= Activation('relu')(xi)
	return xi
"""

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, ksize, stride=1, padding=0, is_act=True) -> None:
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=ksize, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.activation = nn.ReLU()
        self.is_act = is_act
    
    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.is_act == True:
            x = self.activation(x)
        return x
        


class AlprResBlock(nn.Module):
    
    def __init__(self, in_c, out_c, ksize=3, stride=1) -> None:
        super(AlprResBlock, self).__init__()
        self.skip_conv = ConvBlock(in_c=in_c, out_c=out_c, ksize=ksize, stride=stride, padding="same")
        
        self.main_conv = ConvBlock(in_c=out_c, out_c=out_c, ksize=ksize, stride=stride, padding="same", is_act=False)
        self.main_bn = nn.BatchNorm2d(out_c)
        self.main_relu = nn.ReLU()
        
    def forward(self, x):
        identity = x
        x = self.skip_conv(x)
        x = self.main_bn(self.main_conv(x))
        x = torch.add(x, identity)
        x = self.main_relu(x)
        return x