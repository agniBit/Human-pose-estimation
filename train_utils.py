import torch
import torch.nn as nn
import tools
import numpy as np

class Custom_MSELoss(nn.Module):
    def __init__(self):
        super(Custom_MSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')

    def get_dist_error(self,output,target,h=256,w=256):
        x0, y0 = tools.get_preds(output,h,w)
        x1, y1 = tools.get_preds(target,h,w)
        return np.sqrt(np.square(x1-x0)+np.square(y1-y0))

    def forward(self, target, output):
        batch_size, num_joints, h , w = output.shape
        heatmaps_pred = output.reshape((batch_size, num_joints, -1))
        heatmaps_gt = target.reshape((batch_size, num_joints, -1))
        idx0 = heatmaps_pred.argmax(-1).clone().detach()
        idx1 = heatmaps_gt.argmax(-1).clone().detach()
        indices0 = torch.cat(((idx0//h).view(-1, 1), (idx0 % w).view(-1, 1)), dim=1)\
            .reshape(batch_size,num_joints,-1).cpu().numpy()
        indices1 = torch.cat(((idx1//h).view(-1, 1), (idx1 % w).view(-1, 1)), dim=1)\
            .reshape(batch_size,num_joints,-1).cpu().numpy()
        heatmaps_pred = heatmaps_pred.split(1,1)
        heatmaps_gt = heatmaps_gt.split(1, 1)
        loss = 0
        dist_error = np.sqrt(np.square(indices0[:, :, 0] - indices1[:, :, 0]) +
                             np.square(indices0[:, :, 1] - indices1[:, :, 1]))
        for batch in range(batch_size):
            for idx in range(num_joints):
                heatmap_pred = heatmaps_pred[idx].squeeze()
                heatmap_gt = heatmaps_gt[idx].squeeze()
                loss += dist_error[batch][idx] *self.criterion(heatmap_pred, heatmap_gt)
        return loss / num_joints


class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))
                            


class J_MSELoss(nn.Module):
  def __init__(self):
    super(J_MSELoss, self).__init__()
    self.criterion = nn.MSELoss(reduction='mean')

  def forward(self, output, target):
    batch_size = output.size(0)
    num_joints = output.size(1)
    heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
    heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
    loss = 0
    for idx in range(num_joints):
      heatmap_pred = heatmaps_pred[idx].squeeze()
      heatmap_gt = heatmaps_gt[idx].squeeze()
      loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
    return loss


def cyclical_lr(stepsize, min_lr=.00005, max_lr=.01):
    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda