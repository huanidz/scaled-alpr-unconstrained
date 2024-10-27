import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import StanfordCars

from components.model.AlprModel import AlprModel
from datasets import load_dataset
from tqdm import tqdm

torch.set_float32_matmul_precision('high')

data = StanfordCars('./data_stanford_cars', download=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.fc = nn.Linear(64, 128)
        self.classifier = nn.Linear(128, num_classes)

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
    
optimizer = optim.Adam(model.parameters(), lr=0.0003)
criterion = nn.CrossEntropyLoss()
train_dataset = ImageNetDataset(data["train"])
train_loader = DataLoader(train_dataset, batch_size=256, num_workers=8, shuffle=True)
test_loader = DataLoader(data["val"], batch_size=256, num_workers=8, shuffle=False)

epochs = 10
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