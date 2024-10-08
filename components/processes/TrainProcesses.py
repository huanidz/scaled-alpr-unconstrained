import torch
import numpy as np

from components.processes.InferenceProcess import reconstruct
from components.metrics.evaluation import calculate_metrics

from tqdm import tqdm

def evaluate(model, dataloader, eval_threshold, device):
    max_plates = 2
    model.eval()
    max_plates = 2
    with torch.no_grad():
        ious = []
        f1s = []
        for batch_idx, (resized_image, model_input, gt_plate_poly) in tqdm(enumerate(dataloader), desc="Evaluating model..."):
            model_input = model_input.to(device)
            probs, bbox = model(model_input)
            concat_predict_output = torch.cat([probs, bbox], dim=1).detach().cpu()
            for i in range(len(concat_predict_output)):
                results = reconstruct(resized_image[i], concat_predict_output[i], eval_threshold)
                # Calculate metrics
                if len(results) == 0 or len(results) > max_plates:
                    iou, f1 = 0, 0
                else:
                    single_predict_plate_poly = results[0][0].numpy().transpose((1, 0))
                    single_gt_plate_poly = gt_plate_poly[i].numpy()
                    iou, f1 = calculate_metrics(single_predict_plate_poly, single_gt_plate_poly)

                ious.append(iou)
                f1s.append(f1)

        mean_iou = np.mean(ious)
        mean_f1 = np.mean(f1s)
            
    return mean_iou, mean_f1
            
            