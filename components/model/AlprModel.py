import torch.nn as nn
from ..extractor import AlprResBlock, ConvBlock


class AlprModel(nn.Module):
    def __init__(self, scale="base") -> None:
        super(AlprModel, self).__init__()
        supported_scales = ['small','base','large']
        
        if scale not in supported_scales:
            supported_scales_str = ', '.join(supported_scales)
            raise NotImplementedError(f"Scale '{scale}' is currently not supported. Please choose one of these scales: {supported_scales_str}")
                
        # Input's shape: BCHW (batch, channel, height, width)
        # Input's channel must be 3
        if scale == 'base':
            self.conv_batch_1 = ConvBlock(in_c=3, out_c=16, ksize=3, stride=1, padding="same")
            self.conv_batch_2 = ConvBlock(in_c=16, out_c=16, ksize=3, stride=1, padding="same")
            self.downsample_conv_1 = ConvBlock(in_c=16, out_c=16, ksize=3, stride=2, padding=1) # Size = Input/2 (Replace the MaxPool)
            
            self.conv_batch_3 = ConvBlock(in_c=16, out_c=32, ksize=3, stride=1, padding="same")
            self.res_block_1 = AlprResBlock(in_c=32, out_c=32, ksize=3, stride=1)
            
            self.downsample_conv_2 = ConvBlock(in_c=32, out_c=32, ksize=3, stride=2, padding=1) # Size = Input/4 (Replace the MaxPool)
            self.conv_batch_4 = ConvBlock(in_c=32, out_c=64, ksize=3, stride=1, padding="same")
            self.res_block_2 = AlprResBlock(in_c=64, out_c=64, ksize=3, stride=1)
            self.res_block_3 = AlprResBlock(in_c=64, out_c=64, ksize=3, stride=1)
            
            self.downsample_conv_3 = ConvBlock(in_c=64, out_c=64, ksize=3, stride=2, padding=1) # Size = Input/8 (Replace the MaxPool)
            self.conv_batch_5 = ConvBlock(in_c=64, out_c=64, ksize=3, stride=1, padding="same")
            self.res_block_4 = AlprResBlock(in_c=64, out_c=64, ksize=3, stride=1)
            self.res_block_5 = AlprResBlock(in_c=64, out_c=64, ksize=3, stride=1)
            
            self.downsample_conv_4 = ConvBlock(in_c=64, out_c=64, ksize=3, stride=2, padding=1) # Size = Input/16 (Replace the MaxPool)
            self.conv_batch_6 = ConvBlock(in_c=64, out_c=128, ksize=3, stride=1, padding="same")
            self.res_block_6 = AlprResBlock(in_c=128, out_c=128, ksize=3, stride=1)
            self.res_block_7 = AlprResBlock(in_c=128, out_c=128, ksize=3, stride=1)
            self.res_block_8 = AlprResBlock(in_c=128, out_c=128, ksize=3, stride=1)
            self.res_block_9 = AlprResBlock(in_c=128, out_c=128, ksize=3, stride=1)
            
            self.out_probs = ConvBlock(in_c=128, out_c=2, ksize=3, stride=1, padding=1, is_act=False)
            self.out_bbox = ConvBlock(in_c=128, out_c=6, ksize=3, stride=1, padding=1, is_act=False)
            
        
    def forward(self, x):
        x = self.conv_batch_1(x)
        x = self.conv_batch_2(x)
        x = self.downsample_conv_1(x)
        
        x = self.conv_batch_3(x)
        x = self.res_block_1(x)
        x = self.downsample_conv_2(x)
        
        x = self.conv_batch_4(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)
        x = self.downsample_conv_3(x)
        
        x = self.conv_batch_5(x)
        x = self.res_block_4(x)
        x = self.res_block_5(x)
        x = self.downsample_conv_4(x)
        
        x = self.conv_batch_6(x)
        x = self.res_block_6(x)
        x = self.res_block_7(x)
        x = self.res_block_8(x)
        x = self.res_block_9(x)
        
        probs = self.out_probs(x)
        bbox = self.out_bbox(x)
        
        return probs, bbox
        
        
        
            
            
            