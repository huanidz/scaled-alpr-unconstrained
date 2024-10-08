import torch
import torch.nn as nn
from ..extractor import AlprResBlock, ConvBlock
import timm

class AlprBackbone(nn.Module):
    def __init__(self, scale="base") -> None:
        super(AlprBackbone, self).__init__()
        supported_scales = ['tiny','small','base','large']
        
        if scale not in supported_scales:
            supported_scales_str = ', '.join(supported_scales)
            raise NotImplementedError(f"Scale '{scale}' is currently not supported. Please choose one of these scales: {supported_scales_str}")

        if scale == 'base':
            res_block_scales = [32, 64, 64, 128]
        elif scale == 'large':
            res_block_scales = [32, 64, 128, 256]
        elif scale == 'small':
            res_block_scales = [32, 32, 32, 64]
        elif scale == 'tiny':
            res_block_scales = [32, 32, 32, 32]
        else:
            raise NotImplementedError(f"Scale '{scale}' is currently not supported. Please choose one of these scales: {supported_scales_str}")

        self.conv_batch_1 = ConvBlock(in_c=3, out_c=16, ksize=3, stride=1, padding="same")
        self.conv_batch_2 = ConvBlock(in_c=16, out_c=16, ksize=3, stride=1, padding="same")
        self.downsample_conv_1 = ConvBlock(in_c=16, out_c=16, ksize=3, stride=2, padding=1) # Size = Input/2 (Replace the MaxPool)
        
        self.conv_batch_3 = ConvBlock(in_c=16, out_c=res_block_scales[0], ksize=3, stride=1, padding="same")
        self.res_block_1 = self._make_sequence_res_block(num_blocks=1, in_c=res_block_scales[0], out_c=res_block_scales[0], ksize=3, stride=1)
        
        self.downsample_conv_2 = ConvBlock(in_c=res_block_scales[0], out_c=res_block_scales[0], ksize=3, stride=2, padding=1) # Size = Input/4 (Replace the MaxPool)
        self.conv_batch_4 = ConvBlock(in_c=res_block_scales[0], out_c=res_block_scales[1], ksize=3, stride=1, padding="same")
        self.res_block_2_to_3 = self._make_sequence_res_block(num_blocks=2, in_c=res_block_scales[1], out_c=res_block_scales[1], ksize=3, stride=1)
        
        self.downsample_conv_3 = ConvBlock(in_c=res_block_scales[1], out_c=res_block_scales[2], ksize=3, stride=2, padding=1) # Size = Input/8 (Replace the MaxPool)
        self.conv_batch_5 = ConvBlock(in_c=res_block_scales[2], out_c=res_block_scales[2], ksize=3, stride=1, padding="same")
        self.res_block_4_to_5 = self._make_sequence_res_block(num_blocks=2, in_c=res_block_scales[2], out_c=res_block_scales[2], ksize=3, stride=1)
        
        self.downsample_conv_4 = ConvBlock(in_c=res_block_scales[2], out_c=res_block_scales[2], ksize=3, stride=2, padding=1) # Size = Input/16 (Replace the MaxPool)
        self.conv_batch_6 = ConvBlock(in_c=res_block_scales[2], out_c=res_block_scales[3], ksize=3, stride=1, padding="same")
        self.res_block_6_to_9 = self._make_sequence_res_block(num_blocks=4, in_c=res_block_scales[3], out_c=res_block_scales[3], ksize=3, stride=1)
        
    def _make_sequence_res_block(self, num_blocks, in_c, out_c, ksize, stride):
        if num_blocks == 1:
            return AlprResBlock(in_c=in_c, out_c=out_c, ksize=ksize, stride=stride)
        layers = []
        for _ in range(num_blocks):
            layers.append(AlprResBlock(in_c=in_c, out_c=out_c, ksize=ksize, stride=stride))
            in_c = out_c
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv_batch_1(x)
        x = self.conv_batch_2(x)
        x = self.downsample_conv_1(x)
        
        x = self.conv_batch_3(x)
        x = self.res_block_1(x)
        
        
        x = self.downsample_conv_2(x)
        x = self.conv_batch_4(x)
        x = self.res_block_2_to_3(x)
        
        
        x = self.downsample_conv_3(x)
        x = self.conv_batch_5(x)
        x = self.res_block_4_to_5(x)
        
        
        x = self.downsample_conv_4(x)
        x = self.conv_batch_6(x)
        x = self.res_block_6_to_9(x)
        return x

class AlprHead(nn.Module):
    def __init__(self, in_c) -> None:
        super(AlprHead, self).__init__()
        self.out_probs = ConvBlock(in_c=in_c, out_c=2, ksize=3, stride=1, padding=1, is_act=False)
        self.out_bbox = ConvBlock(in_c=in_c, out_c=6, ksize=3, stride=1, padding=1, is_act=False)

    def forward(self, x):
        probs = self.out_probs(x)
        bboxes = self.out_bbox(x)
        return probs, bboxes

class AlprModel(nn.Module):
    def __init__(self, feature_extractor="original", scale="base") -> None:
        super(AlprModel, self).__init__()
        supported_scales = ['tiny','small','base','large']
        supported_feature_extractors = ['original', 'dla']
        
        if scale == 'base':
            res_block_scales = [32, 64, 64, 128]
        elif scale == 'large':
            res_block_scales = [32, 64, 128, 256]
        elif scale == 'small':
            res_block_scales = [32, 32, 32, 64]
        elif scale == 'tiny':
            res_block_scales = [32, 32, 32, 32]

        if scale not in supported_scales:
            supported_scales_str = ', '.join(supported_scales)
            raise NotImplementedError(f"Scale '{scale}' is currently not supported. Please choose one of these scales: {supported_scales_str}")

        self.adapter = None

        if feature_extractor == "original":
            self.backbone = AlprBackbone(scale=scale)
        elif feature_extractor == "dla":
            from_timm = timm.create_model('dla34', pretrained=True)
            self.backbone = nn.Sequential(*list(from_timm.children())[:-5]) # 1, 256, H//16, W//16
            self.adapter = ConvBlock(in_c=256, out_c=res_block_scales[3], ksize=3, stride=1, padding="same")
        
        self.head = AlprHead(in_c=res_block_scales[3])

    def forward(self, x):
        x = self.backbone(x)
        if self.adapter is not None:
            x = self.adapter(x)
        probs, bbox = self.head(x)

        return probs, bbox
        
        
        
            
            
            