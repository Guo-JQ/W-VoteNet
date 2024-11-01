# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from pointnet2_modules import PointnetSAModuleVotes
import pointnet2_utils
from objectness_module import ObjectnessModule
from srm_module import SRModule

def get_index(objectness_label):
    a = torch.arange(objectness_label.shape[1]).unsqueeze(0).repeat((objectness_label.shape[0], 1))

    obj_all_ind = a[objectness_label == 1]  # 所有的index在一个tensor里
    sum_0 = torch.sum(objectness_label, dim=1).cuda().long()
    start = 0
    _idx_batch = torch.zeros((1, 256)).cuda().long()
    for i, num in enumerate(sum_0):
        tmp = obj_all_ind[start:start + num].cuda().long()
        tensor256 = torch.arange(0, 256, dtype=torch.int).cuda().long()
        tensor256[:num] = tmp
        tensor256 = tensor256.unsqueeze(0)  # (1,256)
        _idx_batch = torch.cat((_idx_batch, tensor256), dim=0)
        start += num
    _idx_batch = _idx_batch[1:, :]
    return _idx_batch, sum_0


def decode_scores(net, end_points, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
    net_transposed = net.transpose(2, 1)  # (batch_size, 1024, ..)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    base_xyz = end_points['aggregated_vote_xyz']  # (batch_size, num_proposal, 3)
    center = base_xyz + net_transposed[:, :, 0:3]  # (batch_size, num_proposal, 3)
    end_points['center'] = center

    heading_scores = net_transposed[:, :, 3:3 + num_heading_bin]  # num_heading_bin (Scannet:1)
    heading_residuals_normalized = net_transposed[:, :, 3 + num_heading_bin:3 + num_heading_bin * 2]
    end_points['heading_scores'] = heading_scores  # Bxnum_proposalxnum_heading_bin
    end_points[
        'heading_residuals_normalized'] = heading_residuals_normalized  # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    end_points['heading_residuals'] = heading_residuals_normalized * (
            np.pi / num_heading_bin)  # Bxnum_proposalxnum_heading_bin

    size_scores = net_transposed[:, :, 3 + num_heading_bin * 2:3 + num_heading_bin * 2 + num_size_cluster]
    size_residuals_normalized = net_transposed[:, :,
                                3 + num_heading_bin * 2 + num_size_cluster:3 + num_heading_bin * 2 + num_size_cluster * 4].view(
        [batch_size, num_proposal, num_size_cluster, 3])  # Bxnum_proposalxnum_size_clusterx3
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    end_points['size_residuals'] = size_residuals_normalized * torch.from_numpy(
        mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)

    sem_cls_scores = net_transposed[:, :,
                     3 + num_heading_bin * 2 + num_size_cluster * 4:]  # Bxnum_proposalx10  Scannet:18(num_class)
    end_points['sem_cls_scores'] = sem_cls_scores
    return end_points


class ProposalModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling,
                 seed_feat_dim=256, relation_pair=3, relation_type=['semantic_relation'], random=False):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes(
            npoint=self.num_proposal,
            radius=0.3,
            nsample=16,
            mlp=[self.seed_feat_dim, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True
        )

        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(128 + 128, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 2 + 3 + num_heading_bin * 2 + num_size_cluster * 4 + self.num_class, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

        self.srm = SRModule(relation_pair=relation_pair, relation_type=relation_type, random=random)
        self.objm = ObjectnessModule(128)

    def forward(self, xyz, features, end_points):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds = self.vote_aggregation(xyz,features)  # torch.Size([8, 256, 3]) torch.Size([8, 128, 256]) torch.Size([8, 256])
            sample_inds = fps_inds
        elif self.sampling == 'seed_fps':
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = end_points['seed_xyz'].shape[1]
            batch_size = end_points['seed_xyz'].shape[0]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            log_string('Unknown sampling strategy: %s. Exiting!' % (self.sampling))
            exit()
        end_points['aggregated_vote_xyz'] = xyz  # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds  # (batch_size, num_proposal,)

        # --------- PROPOSAL GENERATION ---------        
        objectness_pred, end_points = self.objm(features, end_points)
        idx_obj, sum_one = get_index(objectness_pred)
        end_points['idx_obj'] = idx_obj
        end_points['sum_one'] = sum_one

        end_points = self.srm(features, end_points)
        rn_feature = end_points['rn_feature']

        features = torch.cat((features, rn_feature), 1)  # torch.Size([8, 256, 256])

        net = F.relu(self.bn1(self.conv1(features)))  # torch.Size([8, 128, 256])
        net = F.relu(self.bn2(self.conv2(net)))
        net = self.conv3(net)  # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)

        end_points = decode_scores(net, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster,self.mean_size_arr)

        return end_points


if __name__ == '__main__':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import DC

    net = ProposalModule(DC.num_class, DC.num_heading_bin,
                         DC.num_size_cluster, DC.mean_size_arr,
                         128, 'seed_fps').cuda()
    end_points = {'seed_xyz': torch.rand(8, 1024, 3).cuda()}
    out = net(torch.rand(8, 1024, 3).cuda(), torch.rand(8, 256, 1024).cuda(), end_points)
    for key in out:
        print(key, out[key].shape)
