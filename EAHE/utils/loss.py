import torch.nn as nn
import torch
import torch.nn.functional as F

triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=1, reduce=False,size_average=False)
mse = torch.nn.MSELoss()
class get_edge_warped(nn.Module):
    def __init__(self, *args, **kwargs):
        super(get_edge_warped, self).__init__()

    def forward(self, target):
        boundary_targets = F.conv2d(target.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0
        return boundary_targets

class ComputeAlignLoss:
    def __init__(self, model):
        super(ComputeAlignLoss, self).__init__()
        self.hyp = {'lr0': 0.0001, 'lrf': 0.1, 'momentum': 0.9, 'weight_decay': 0.0, 'warmup_epochs': 0.0,
                    'warmup_momentum': 0.8,
                    'warmup_bias_lr': 0.1, 'loss_scale1': 16.0, 'loss_scale2': 4.0, 'loss_scale3': 1.0, 'hsv_h': 0.015,
                    'hsv_s': 0.2,
                    'hsv_v': 0.15, 'degrees': 0.05, 'translate': 0.1, 'scale': 0.1, 'shear': 0.01,
                    'perspective': 0.0005, 'flipud': 0.1, 'fliplr': 0.5}
        self.scales = [self.hyp['loss_scale1'], self.hyp['loss_scale2'], self.hyp['loss_scale3']]
        self.scales1 = [1., 1., 1.]
        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)
    def __call__(self, pred, imgs, images, img1_warped_mask, offset):  # for consistency
        eps = 0.01
        warped_imgs, warped_ones = pred[1:3]
        warp_edge = pred[3]
        offset_s = pred[0]
        target_image, target_mask = imgs[:, :3, ...], imgs[:, 3:4, ...]
        target_egde = img1_warped_mask
        target_mask = (target_mask > eps).expand(-1, 3, -1, -1)
        bs, device = images.shape[0], images.device
        loss_per_level = [torch.zeros(1, device=device) for _ in range(3)]
        loss_per_level3 = [torch.zeros(1, device=device) for _ in range(3)]
        for i, warped_img in enumerate(warped_imgs):
            warped_mask = (warped_ones[i] > eps).expand(-1, 3, -1, -1)
            if warped_mask.sum() == 0:
                # return None, None
                continue
            overlap_mask = target_mask & warped_mask
            loss_per_level3[i] += F.binary_cross_entropy_with_logits(warp_edge[i] * overlap_mask,
                                                                     target_egde * overlap_mask,
                                                                     reduce='none')
            loss_per_level[i] += F.l1_loss(warped_img*overlap_mask, target_image*overlap_mask)
        loss = sum([scale * loss_per_level[i] for i, scale in enumerate(self.scales1)])
        return loss * bs, torch.cat((*loss_per_level, loss)).detach(), torch.cat((*loss_per_level3, loss)).detach()


