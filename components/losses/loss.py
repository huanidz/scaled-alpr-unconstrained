import torch
import torch.nn as nn

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
    
class AlprLoss(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.log_loss = LogLoss()
        self.l1_loss = L1Loss()
        
    def forward(self, predict, target):
        
        # Prob's shape: B x 2 x Out_H x Out_W (simplied: Bx2xHxW) (Out_H Out_W are depends on the input size)
        # Bbox's shape: B x 6 x Out_H x Out_W
        
        # Predict's shape: Concat(Prob, Bbox) --> B x 8 x Out_H x Out_W
        
        B, C, H, W = target.shape
        
        obj_prob_predict = predict[:, 0] # BxHxW
        obj_prob_target = target[:, 0] # BxHxW
        
        non_obj_prob_predict = 1 - predict[:, 0] # BxHxW
        non_obj_prob_target = target[:, 0] # BxHxW

        affine_predict = predict[:, 2:] # Bx6xHxW
        pts_true = target[:, 1:] # ??
        
        affinex = torch.stack([torch.maximum(affine_predict[:, 0], 0.), affine_predict[:, 1], affine_predict[:, 2]], dim=1)
        affiney = torch.stack([affine_predict[:, 3], torch.maximum(affine_predict[:, 4], 0),affine_predict[:, 5]], dim=1)
                
        v = torch.Tensor([0.5])
        base = (-v).expand(B, H, W, 4).reshape(B, H, W, 4, 3).permute(0, 3, 1, 2)
        pts = torch.zeros_like(base)
        
        for i in range(0, 12, 3):
            row = base[..., i:(i+3)]
            ptsx = torch.sum(affinex.permute(0, 2, 3, 1) * row, dim=3)
            ptsy = torch.sum(affiney.permute(0, 2, 3, 1) * row, dim=3)

            pts_xy = torch.stack([ptsx, ptsy], dim=3).permute(0, 3, 1, 2)
            pts = torch.cat([pts, pts_xy], dim=1)
            
        flags = obj_prob_target.reshape(B, 1, H, W)
        
        loss = self.l1_loss(pts_true*flags, pts*flags) + self.log_loss(obj_prob_predict, obj_prob_target) + self.log_loss(non_obj_prob_predict, non_obj_prob_target)
        return loss        
        
    