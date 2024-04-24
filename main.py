import torch
torch.manual_seed(11)
from torch.utils.data import DataLoader
from components.losses.loss import AlprLoss
from components.data.AlprData import AlprDataset
from components.model.AlprModel import AlprModel

from utils.util_func import count_parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

model = AlprModel().to(device)
print(f'The model has {count_parameters(model):,} trainable parameters')

criteria = AlprLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

dataset = AlprDataset(images_folder="./train_data/WPOD_Training_Dataset/images", labels_folder="./train_data/WPOD_Training_Dataset/labels", input_size=384)

batch_size = 8

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

epochs = 200
lowest_loss = 999999

print("Start training..")
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    print("-" * 10)
    running_loss = 0.
    last_loss = 0.
    
    for batch_idx, (image, output_feature_map) in enumerate(train_loader):
        
        optimizer.zero_grad()
        
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
            if loss_per_batch < lowest_loss:
                lowest_loss = loss_per_batch
                
    # Save the model each epoch
    print("Saving model...")
    torch.save(model.state_dict(), f"model_{epoch}.pth")
    
