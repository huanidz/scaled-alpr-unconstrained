import torch
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser 

from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm

torch.set_float32_matmul_precision('high')

data = load_dataset("evanarlian/imagenet_1k_resized_256")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = ArgumentParser()
args.add_argument("--scale", type=str, default="base")
args.add_argument("--lr", type=float, default=1e-4)
args.add_argument("--bs", type=int, default=16) 
args.add_argument("--epochs", type=int, default=200)
args.add_argument("--nwk", type=int, default=8)
args = args.parse_args()

supported_scales = ['tiny','small','base','large']
scale_out_dim = [32, 64, 128, 256]
scale_out_dim_dict = dict(zip(supported_scales, scale_out_dim))

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

class AlprModel(nn.Module):
    def __init__(self, scale="base") -> None:
        super(AlprModel, self).__init__()
        supported_scales = ['tiny','small','base','large']
        
        if scale not in supported_scales:
            supported_scales_str = ', '.join(supported_scales)
            raise NotImplementedError(f"Scale '{scale}' is currently not supported. Please choose one of these scales: {supported_scales_str}")
                
        # Input's shape: BCHW (batch, channel, height, width)
        # Input's channel must be 3
        if scale == 'base':
            res_block_scales = [32, 64, 64, 128]
        elif scale == 'large':
            res_block_scales = [32, 64, 128, 256]
        elif scale == 'small':
            res_block_scales = [32, 32, 32, 64]
        elif scale == 'tiny':
            res_block_scales = [32, 32, 32, 32]


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
        
        self.out_probs = ConvBlock(in_c=res_block_scales[3], out_c=2, ksize=3, stride=1, padding=1, is_act=False)
        self.out_bbox = ConvBlock(in_c=res_block_scales[3], out_c=6, ksize=3, stride=1, padding=1, is_act=False)
    
   
    
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
        
        probs = torch.softmax(self.out_probs(x), dim=1) # B, 2, H, W
        bbox = self.out_bbox(x)
        
        return probs, bbox
    


base_model = AlprModel(scale="small")
backbone = nn.Sequential(*list(base_model.children())[:-2])

class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        data = self.dataset[index]
        image = data["image"]
        label = data["label"]
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)
 
class Pretraining(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(scale_out_dim_dict[args.scale], scale_out_dim_dict[args.scale] * 2)
        self.classifier = nn.Linear(scale_out_dim_dict[args.scale] * 2, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.classifier(x)
        return x 
    
model = Pretraining(backbone, 1000)
model.to(device)
model = torch.compile(model)
    
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()
train_dataset = ImageNetDataset(data["train"])
train_loader = DataLoader(train_dataset, batch_size=args.bs, num_workers=args.nwk, shuffle=True)
test_loader = DataLoader(data["val"], batch_size=args.bs, num_workers=args.nwk, shuffle=False)

epochs = args.epochs
for epoch in range(epochs):
    model.train()
    print(f"Epoch {epoch + 1}/{epochs}")
    running_loss = 0.0
    for images, labels in (progress := tqdm(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress.set_description(f"Loss: {running_loss}")
        running_loss = 0.0
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {correct / total}")
    
    # Save each epoch    
    torch.save(model.state_dict(), f"pretrained_alpr_epoch_{epoch + 1}.pt")