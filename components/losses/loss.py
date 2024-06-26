import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LogLoss(nn.Module):
    def __init__(self, eps=1e-10):
        super(LogLoss, self).__init__()
        self.eps = eps
    
    def forward(self, predict, target):
        B, C, H, W = target.shape  # target shape: (B, C, H, W)

        predict = torch.clamp(predict, self.eps, 1.0)  # (B, C, H, W)
        predict = -torch.log(predict)  # (B, C, H, W) 
        predict = torch.mul(predict, target)  # Element-wise multiplication, (B, C, H, W)
        predict = predict.view(B, C * H * W)  # Reshape/flatten last three dimensions, (B, C * H * W)
        predict = torch.sum(predict, dim=1)  # Sum over C*H*W, output shape: (B,)

        return predict
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, eps=1e-10):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
    
    def forward(self, predict, target):
        B, C, H, W = target.shape  # target shape: (B, C, H, W)
        
        # Ensure predictions are within (eps, 1-eps) to avoid log(0)
        predict = torch.clamp(predict, self.eps, 1.0 - self.eps)
        
        # Compute the focal loss
        pt = predict * target + (1 - predict) * (1 - target)  # pt = P if target == 1 else 1 - P
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = -focal_weight * (target * torch.log(predict) + (1 - target) * torch.log(1 - predict))
        
        # Sum over all elements
        loss = loss.view(B, C * H * W)
        loss = torch.sum(loss, dim=1)  # Sum over C*H*W, output shape: (B,)

        return loss

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
    
    def forward(self, predict, target):
        B, C, H, W = target.shape  # target shape: (B, C, H, W)
        
        res = predict - target  # Subtraction, res shape: (B, C, H, W) 
        res = res.abs()  # Taking absolute, res shape: (B, C, H, W)
        res = res.view(B, C * H * W)  # Reshape/flatten last three dimensions, res shape: (B, C*H*W)
        res = torch.sum(res, dim=1)  # Sum over C*H*W, res shape: (B,) 
        return res  # Return tensor with shape: (B,) 

class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss()
    
    def forward(self, predict, target):
        return self.smooth_l1_loss(predict, target)

class AlprLoss(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.non_vs_object_loss = FocalLoss()
        self.pts_regression_loss = SmoothL1Loss()
                
    def forward(self, predict, target):
        
        # Prob's shape: B x 2 x Out_H x Out_W (simplied: Bx2xHxW) (Out_H Out_W are depends on the input size)
        # Bbox's shape: B x 6 x Out_H x Out_W
        
        # Predict's shape: Concat(Prob, Bbox) --> B x 8 x Out_H x Out_W
        
        B, C, H, W = target.shape
        
        obj_prob_predict = predict[:, 0].unsqueeze(1) # Bx1xHxW
        obj_prob_target = target[:, 0].unsqueeze(1) # Bx1xHxW
        
        non_obj_prob_predict = predict[:, 1].unsqueeze(1) # Bx1xHxW
        non_obj_prob_target = 1.0 - target[:, 0].unsqueeze(1) # Bx1xHxW

        affine_predict = predict[:, 2:] # Bx6xHxW
        pts_true = target[:, 1:] # Bx8xHxW
        
        
        affinex = torch.stack([torch.maximum(affine_predict[:, 0], torch.zeros_like(affine_predict[:, 0])), affine_predict[:, 1], affine_predict[:, 2]], dim=1)
        
        affiney = torch.stack([affine_predict[:, 3], torch.maximum(affine_predict[:, 4], torch.zeros_like(affine_predict[:, 4])),affine_predict[:, 5]], dim=1)
        
        v = 0.5
        base = torch.tensor([[[[-v, -v, 1., v, -v, 1., v, v, 1., -v, v, 1.]]]], device=predict.device)
        base = base.repeat(B, H, W, 1)
        pts = torch.zeros(B, 0, H, W, device=predict.device)
        
        for i in range(0, 12, 3):
            row = base[..., i:(i+3)]
            ptsx = torch.sum(affinex.permute(0, 2, 3, 1) * row, dim=3)
            ptsy = torch.sum(affiney.permute(0, 2, 3, 1) * row, dim=3)

            pts_xy = torch.stack([ptsx, ptsy], dim=3).permute(0, 3, 1, 2)
            pts = torch.cat([pts, pts_xy], dim=1)
            
        flags = obj_prob_target.reshape(B, 1, H, W)
        
        l1_loss = self.pts_regression_loss(pts*flags, pts_true*flags)
        obj_log_loss = self.non_vs_object_loss(obj_prob_predict, obj_prob_target)
        non_obj_log_loss = self.non_vs_object_loss(non_obj_prob_predict, non_obj_prob_target)
        
        loss = 2 * l1_loss + obj_log_loss + 0.5 * non_obj_log_loss        
        loss = torch.mean(loss)
        
        return loss        
        
    