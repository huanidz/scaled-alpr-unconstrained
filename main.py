import torch
torch.manual_seed(11)
from components.model.AlprModel import AlprModel
from components.losses.loss import AlprLoss
from components.data.AlprData import AlprDataset
from torch.utils.data import DataLoader
from time import perf_counter

from utils.util_func import count_parameters

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


model = AlprModel().to(device)
print(f'The model has {count_parameters(model):,} trainable parameters')

criteria = AlprLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0003)

dataset = AlprDataset(images_folder="./train_data/images", labels_folder="./train_data/labels", input_size=208)

train_loader = DataLoader(dataset, batch_size=3, shuffle=False)

epochs = 200
lowest_loss = 999999

for epoch in range(epochs):
    running_loss = 0.
    last_loss = 0.
    
    for batch_idx, (image, output_feature_map) in enumerate(train_loader):
        
        optimizer.zero_grad()
        
        image = image.to(device)
        # print(f"==>> image: {image}")
        output_feature_map = output_feature_map.to(device)
        # print(f"==>> output_feature_map: {output_feature_map[0][0]}") # 1. 9. 24. 24
        probs, bbox = model(image)
        concat_predict_output = torch.cat([probs, bbox], dim=1)
        loss = criteria(concat_predict_output, output_feature_map)
        
        loss.backward()
        
        optimizer.step()
            
        running_loss += loss.item()
        if batch_idx % 2 == 0:
            last_loss = running_loss / 3 # loss per batch
            if last_loss < lowest_loss:
                lowest_loss = last_loss
            print('  batch {} loss: {}'.format(batch_idx + 1, last_loss))
            print("New lowest loss: ", lowest_loss)
            running_loss = 0.
        
    # break

# Training Loop











# print("Start warming up 100 iteration batch=1..")
# start_time = perf_counter()
# input = torch.randn((16, 3, 384, 384), dtype=torch.float32).to(device)
# probs, bbox = model(input)

# # Simulate the loss backprop
# target_probs = torch.randn((16, 2, 24, 24), dtype=torch.float32).to(device)
# target_bbox = torch.randn((16, 6, 24, 24), dtype=torch.float32).to(device)

# dummy_target_output = torch.randn((16, 9, 24, 24), dtype=torch.float32).to(device)
# concat_predict_output = torch.cat([probs, bbox], dim=1)
# loss = AlprLoss()
# loss_val = loss(concat_predict_output, dummy_target_output)
# loss_val.backward()
# end_time = perf_counter()
# elaps = end_time - start_time
# print(f"Finish warming up. Took {elaps}ms total. Average = {elaps}ms per iteration.")
