import os
import torch
import argparse
from torch.utils.data import DataLoader
from components.losses.loss import AlprLoss
from components.data.AlprData import AlprDataset
from components.model.AlprModel import AlprModel
from utils.util_func import count_parameters

argparser = argparse.ArgumentParser()
argparser.add_argument("--data", help="Path to data folder", default="./train_data", required=True, type=str)
argparser.add_argument("--lr", help="Learning rate", default=3e-4, type=float)
argparser.add_argument("--bs", help="Batch size", default=16, type=int)
argparser.add_argument("--epochs", help="Number of epochs", default=200, type=int)
argparser.add_argument("--size", help="Size of input image", default=384, type=int)
argparser.add_argument("--resume_from", help="Resume from pth checkpoint (path)", default=None, type=str)
args = argparser.parse_args()
print(args)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AlprModel().to(device)
print(f'Alpr model has {count_parameters(model):,} trainable parameters')

criteria = AlprLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)

if args.data[:-1] == "/":
    args.data = args.data[:-1]
    
images_path = args.data + "/images"
labels_path = args.data + "/labels"

dataset = AlprDataset(images_folder=images_path, labels_folder=labels_path, input_size=args.size)

batch_size = args.bs

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

num_epochs = args.epochs

if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")

if args.resume_from:
    load_path = args.resume_from
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded. Resuming from epoch {epoch}...")
else:
    epoch = 0


print("Start training..")
for epoch in range(num_epochs):
    
    # Skipping epoch if resume
    if epoch == 0 and args.resume_from:
        continue
    
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("-" * 10)
    running_loss = 0.0
    
    for batch_idx, (image, output_feature_map) in enumerate(train_loader):
        
        optimizer.zero_grad(set_to_none=True)
        
        image = image.to(device)
        output_feature_map = output_feature_map.to(device)
        probs, bbox = model(image)
        concat_predict_output = torch.cat([probs, bbox], dim=1)
        loss = criteria(concat_predict_output, output_feature_map)
        
        loss.backward()
        
        optimizer.step()
            
        running_loss += loss.item()
        if batch_idx % (batch_size - 1) == 0 and batch_idx != 0:
            loss_per_batch = running_loss / batch_size
            print(f'  Epoch {epoch+1:03d}, Batch {batch_idx+1:03d}, Loss: {loss_per_batch:.4f}')
            running_loss = 0.0
                
    # Saving each epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"./checkpoints/latest.pth")
    
