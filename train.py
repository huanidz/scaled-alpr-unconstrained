import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from components.losses.loss import AlprLoss
from components.data.AlprData import AlprDataset
from components.model.AlprModel import AlprModel
from components.processes.InferenceProcess import reconstruct
from components.metrics.evaluation import calculate_metrics
from utils.util_func import count_parameters

argparser = argparse.ArgumentParser()
argparser.add_argument("--data", help="Path to data folder", default="./train_data", required=True, type=str)
argparser.add_argument("--lr", help="Learning rate", default=3e-4, type=float)
argparser.add_argument("--bs", help="Batch size", default=16, type=int)
argparser.add_argument("--epochs", help="Number of epochs", default=200, type=int)
argparser.add_argument("--eval_after", help="Evaluate model after n epochs", default=1, type=int)
argparser.add_argument("--size", help="Size of input image", default=384, type=int)
argparser.add_argument("--scale", help="Scale of the model (tiny, small, base, large)", default="base", type=str)
argparser.add_argument("--resume_from", help="Resume from pth checkpoint (path)", default=None, type=str)
args = argparser.parse_args()
print(args)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AlprModel(scale=args.scale).to(device)
print(f'Alpr model has {count_parameters(model):,} trainable parameters')

criteria = AlprLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)

if args.data[:-1] == "/":
    args.data = args.data[:-1]
    
images_path = args.data + "/images"
labels_path = args.data + "/labels"

dataset = AlprDataset(images_folder=images_path, labels_folder=labels_path, input_size=args.size)
eval_dataset = AlprDataset(images_folder=images_path, labels_folder=labels_path, input_size=args.size, mode="eval")

batch_size = args.bs

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
test_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

num_epochs = args.epochs
n_epochs_eval = args.eval_after
last_eval_epoch = 0

if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")

if args.resume_from:
    load_path = args.resume_from
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_f1_score = checkpoint['best_f1_score']
    last_eval_epoch = epoch
    print(f"Checkpoint loaded. Resuming from epoch {epoch}...")
else:
    epoch = 0
    best_f1_score = 0


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

    # Eval
    if (epoch - last_eval_epoch) % n_epochs_eval == 0:
        print("Evaluating model...")
        model.eval()
        eval_threshold = 0.5
        with torch.no_grad():
            ious = []
            f1s = []
            for batch_idx, (resized_image, model_input, gt_plate_poly) in enumerate(test_loader):
                model_input = model_input.to(device)
                output_feature_map = output_feature_map.to(device)
                probs, bbox = model(model_input)
                concat_predict_output = torch.cat([probs, bbox], dim=1).detach().cpu()
                
                for i in range(len(concat_predict_output)):
                    results = reconstruct(resized_image[i], concat_predict_output[i], eval_threshold)

                    # Calculate metrics
                    if len(results) == 0:
                        iou, f1 = 0, 0
                    else:
                        single_predict_plate_poly = results[0][0].numpy().transpose((1, 0))
                        single_gt_plate_poly = gt_plate_poly[i].numpy()
                        iou, f1 = calculate_metrics(single_predict_plate_poly, single_gt_plate_poly)

                    ious.append(iou)
                    f1s.append(f1)

            mean_iou = np.mean(ious)
            mean_f1 = np.mean(f1s)
            print(f"IoU: {mean_iou:.4f}, F1_Score: {mean_f1:.4f}")
            if mean_f1 > best_f1_score:
                print("New higher F1-Score, saving model...")
                best_f1_score = mean_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1_score': best_f1_score
                }, f"./checkpoints/{args.scale}_best.pth")

        last_eval_epoch = epoch
        model.train()

    # Saving each epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_f1_score': best_f1_score
    }, f"./checkpoints/{args.scale}_latest.pth")
    
