import os
import torch
import argparse
from torch.utils.data import DataLoader
from components.losses.loss import AlprLoss
from components.data.AlprData import AlprDataset
from components.model.AlprModel import AlprModel
from components.processes.TrainProcesses import evaluate
from utils.util_func import count_parameters, init_weights
from time import perf_counter
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR

argparser = argparse.ArgumentParser()
argparser.add_argument("--data", help="Path to data folder", default="./train_data", required=True, type=str)
argparser.add_argument("--lr", help="Learning rate", default=1e-3, type=float)
argparser.add_argument("--bs", help="Batch size", default=16, type=int)
argparser.add_argument("--epochs", help="Number of epochs", default=200, type=int)
argparser.add_argument("--eval_after", help="Evaluate model after n epochs", default=1, type=int)
argparser.add_argument("--size", help="Size of input image", default=384, type=int)
argparser.add_argument("--scale", help="Scale of the model (tiny, small, base, large)", default="base", type=str)
argparser.add_argument("--resume_from", help="Resume from pth checkpoint (path)", default=None, type=str)
argparser.add_argument("--fe", help="type of feature extractor (original or dla)", default="original", type=str)
args = argparser.parse_args()
print(args)

torch.set_float32_matmul_precision("high")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AlprModel(scale=args.scale, feature_extractor=args.fe).to(device)
model.apply(init_weights)
model = torch.compile(model)
print(f'Alpr model has {count_parameters(model):,} trainable parameters')

criteria = AlprLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
# optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5, eps=1e-7, betas=(0.9, 0.999))

if args.data[:-1] == "/":
    args.data = args.data[:-1]
    
images_path = args.data + "/train/images"
labels_path = args.data + "/train/labels"

eval_images_path = args.data + "/eval/images"
eval_labels_path = args.data + "/eval/labels"

dataset = AlprDataset(images_folder=images_path, labels_folder=labels_path, input_size=args.size)

train_eval_dataset = AlprDataset(images_folder=images_path, labels_folder=labels_path, input_size=args.size, mode="eval")
test_eval_dataset = AlprDataset(images_folder=eval_images_path, labels_folder=eval_labels_path, input_size=args.size, mode="eval")

batch_size = args.bs

# Ugly, may refract but it worked!
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
train_eval_loader = DataLoader(train_eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
test_loader = DataLoader(test_eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

num_epochs = args.epochs
n_epochs_eval = args.eval_after
last_eval_epoch = 0
print_per_n_step = 10

if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")

if args.resume_from:
    load_path = args.resume_from
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    last_epoch = checkpoint['epoch']
    best_f1_score = checkpoint['best_f1_score']
    last_eval_epoch = last_epoch
    print(f"Checkpoint loaded. Resuming from epoch {last_epoch}...")
else:
    last_epoch = 0
    best_f1_score = 0

# Cosine annealing learning rate
# scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)
scheduler = CyclicLR(optimizer, base_lr=3e-4, max_lr=1e-3, step_size_up=20, cycle_momentum=False)


print("Start training..")
for epoch in range(num_epochs):
    
    # Skipping epoch if resume
    if args.resume_from and epoch < last_epoch:
        continue
    
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("-" * 10)
    running_loss = 0.0
    time_per_step = []
    
    for batch_idx, (image, output_feature_map) in enumerate(train_loader):
        start = perf_counter()
        
        optimizer.zero_grad(set_to_none=True)
        
        image = image.to(device)
        output_feature_map = output_feature_map.to(device)
        with amp.autocast(dtype=torch.bfloat16):
            probs, bbox = model(image)
            concat_predict_output = torch.cat([probs, bbox], dim=1)
            loss = criteria(concat_predict_output, output_feature_map)
            
        loss.backward()
        # norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        torch.cuda.synchronize()
        
        running_loss += loss.item()
        end = perf_counter()
        total_time_per_step_ms = (end - start) * 1000
        time_per_step.append(total_time_per_step_ms)
        
        if (batch_idx % print_per_n_step == 0 and batch_idx != 0) or batch_idx == len(train_loader) - 1:
            avg_time = sum(time_per_step)
            current_lr = optimizer.param_groups[0]['lr']
            print(f'  Batch {batch_idx+1:03d}, Loss: {running_loss / (print_per_n_step):.4f}, Time: {avg_time:.4f} ms, LR: {current_lr:.4f}')
            time_per_step = []
            running_loss = 0.0
    
    scheduler.step()

    # Eval
    if (epoch - last_eval_epoch) % n_epochs_eval == 0 and epoch != 0:
        print("Evaluating model...")
        train_mean_iou, train_mean_f1 = evaluate(model=model, dataloader=train_eval_loader, eval_threshold=0.5, device=device)
        print(f"[TRAIN] IoU: {train_mean_iou:.4f}, F1_Score: {train_mean_f1:.4f}")
        
        test_mean_iou, test_mean_f1 = evaluate(model=model, dataloader=test_loader, eval_threshold=0.5,  device=device)
        print(f"[TEST ] IoU: {test_mean_iou:.4f}, F1_Score: {test_mean_f1:.4f}")
        
        if test_mean_f1 > best_f1_score:
            print("Higher f1 score found. Saving model...")
            best_f1_score = test_mean_f1
            save_path = f"checkpoints/scale_{args.scale}_size_{args.size}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1_score': best_f1_score
            }, save_path)
            
        last_eval_epoch = epoch
        model.train()

    # Saving each epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_f1_score': best_f1_score
    }, f"./checkpoints/scale_{args.scale}_size_{args.size}_latest.pth")
    
