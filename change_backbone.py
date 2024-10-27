import torch
import torch.nn as nn
from components.model.AlprModel import AlprModel, AlprBackbone, AlprHead, ConvBlock
import timm
from utils.util_func import count_parameters

base_model = AlprBackbone(scale="small")
timm_model = timm.create_model("mobilenetv3_small_100.lamb_in1k", pretrained=True)
head_model = AlprHead(in_c=64)



class AlprAdapter(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        
        self.seq2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        
        self.seq3 = nn.Sequential(
            nn.Conv2d(out_c, out_c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_c * 2),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        return x

class AlprModelTimm(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("mobilenetv3_small_100.lamb_in1k", pretrained=True)
        numparams = count_parameters(self.backbone)
        print(f"Number of parameters: {numparams}")
        
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-5])
        numparams = count_parameters(self.backbone)
        print(f"Number of parameters: {numparams}")
        self.adapter = AlprAdapter(in_c=16, out_c=32)
        self.head = AlprHead(in_c=64)

    def forward(self, x):
        x = self.backbone(x)
        print(f"==>> x.shape: {x.shape}")
        x = self.adapter(x)
        probs, bbox = self.head(x)
        return probs, bbox
