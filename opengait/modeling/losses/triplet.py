import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import BaseLoss, gather_and_scale_wrapper


class TripletLoss(BaseLoss):
    def __init__(self, margin, loss_term_weight=1.0):
        super(TripletLoss, self).__init__(loss_term_weight)
        self.margin = margin
    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        # embeddings: [n, c, p], label: [n]

        embeddings = embeddings.permute(
            2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]

        ref_embed, ref_label = embeddings, labels
        dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]
        mean_dist = dist.mean((1, 2))  # [p]
        ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)
        mean_ap_dist = ap_dist.mean((1,2,3))
        mean_an_dist = an_dist.mean((1,2,3))
        dist_diff = (ap_dist - an_dist).view(dist.size(0), -1)
        loss = F.relu(dist_diff + self.margin)

        hard_loss = torch.max(loss, -1)[0]
        loss_avg, loss_num = self.AvgNonZeroReducer(loss)

        self.info.update({
            'loss': loss_avg.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'loss_num': loss_num.detach().clone(),
            'mean_dist': mean_dist.detach().clone(),
            'mean_ap_dist': mean_ap_dist.detach().clone(),
            'mean_an_dist': mean_an_dist.detach().clone(),})

        return loss_avg, self.info

    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def ComputeDistance(self, x, y):
        """
            x: [p, n_x, c]
            y: [p, n_y, c]
        """
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        return dist

    def Convert2Triplets(self, row_labels, clo_label, dist):
        """
            row_labels: tensor with size [n_r]
            clo_label : tensor with size [n_c]
        """
        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).bool()  # [n_r, n_c]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        p, n, _ = dist.size()
        ap_dist = dist[:, matches].view(p, n, -1, 1)  # [n, p, postive, 1]
        an_dist = dist[:, diffenc].view(p, n, 1, -1)  # [n, p ,1, negative]
        return ap_dist, an_dist



class TripletDouble8124Loss(BaseLoss):
    def __init__(self, margin, loss_term_weight=1.0, start=30000,):
        super(TripletDouble8124Loss, self).__init__(loss_term_weight)
        self.margin = margin
        self.count = 0
        self.start = start
        self.eps = 1e-3

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels, bnn):
        # embeddings: [n, c, p], label: [n], bnn: [p, n, c]
        self.centers = bnn.permute(2, 0, 1).contiguous().float()
        embeddings = embeddings.permute(2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]

        ref_embed, ref_label = embeddings, labels

        distmat = torch.pow(embeddings, 2).sum(dim=2) + torch.pow(self.centers, 2).sum(dim=2) - (
                    2 * embeddings * self.centers).sum(dim=2)  # [p,n]
        distmat = torch.sqrt(F.relu(distmat))
        dist_d = distmat.mean()

        # embeddings ?
        if self.count >= self.start:
            dist = self.ComputeDistance(embeddings, ref_embed)
            mean_dist = dist.mean((1, 2))
            ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist, distmat)  # 8, 124

        else:
            self.count += 1
            if self.count == self.start:
                print("---------------------------starting!---------------------------")
            dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]
            mean_dist = dist.mean((1, 2))  # [p]
            ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)

        dist_diff = (ap_dist - an_dist).view(dist.size(0), -1)  # dist_diff
        dist_d /= 128
        loss = F.relu(dist_diff + self.margin) #+ dist_d)

        hard_loss = torch.max(loss, -1)[0]
        loss_avg, loss_num = self.AvgNonZeroReducer(loss)

        self.info.update({
            'loss': loss_avg.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'center_loss': dist_d.detach().clone(),
            'loss_num': loss_num.detach().clone(),
            'mean_dist': mean_dist.detach().clone()})

        return loss_avg, self.info

    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def ComputeDistance(self, x, y):
        """
            x: [p, n_x, c] embeddings
            y: [p, n_y, c]
        """

        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        return dist
    def Convert2Triplets(self, row_labels, clo_label, dist, dist_d=None):
        """
            row_labels: tensor with size [n_r]
            clo_label : tensor with size [n_c]
        """
        #
        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).bool()  # [n_r, n_c] [128, 128]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        p, n, _ = dist.size()  # [p, n, n]
        ap_dist = dist[:, matches].view(p, n, -1, 1)  # [p, n, 4, 1]
        an_dist = dist[:, diffenc].view(p, n, 1, -1)  # [p, n, 1, 124]
        if dist_d != None:
            ap_dist = torch.cat([ap_dist, dist_d.reshape(p,n,1,1)], dim=2)  # # [p, n, 5, 1]
        return ap_dist, an_dist
