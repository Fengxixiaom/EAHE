import argparse
import logging
import sys
from pathlib import Path
from itertools import product
sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)
from models.common import *
from models.experimental import *
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import fuse_conv_and_bn, model_info, initialize_weights, \
    select_device, check_anomaly
from utils.stitching import DLTSolver, STN

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

autocast = torch.cuda.amp.autocast

class CorrNeigh(nn.Module):
    def __init__(self, kernelSize):
        super(CorrNeigh, self).__init__()
        assert kernelSize%2==1
        self.kernelSize=kernelSize
        self.paddingSize=kernelSize//2
        self.padding=torch.nn.ZeroPad2d(self.paddingSize)

    def do_forward(self, x, y):
        ## x, y should be normalized
        n, c, w, h=x.size()
        coef=[]
        y = self.padding(y)
        for i, j in product(range(self.kernelSize), range(self.kernelSize)):
            coef.append(torch.sum(x*y.narrow(2, i, w).narrow(3, j, h), dim=1, keepdim=True))
        coef=torch.cat(coef, dim=1)
        return coef

    def forward(self, x, y):

        if self.training:
            coef=self.do_forward(x, y)
        else:
            with torch.no_grad():
                coef=self.do_forward(x, y)
        return coef

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

class HEstimator(nn.Module):
    def __init__(self, input_size=128, strides=(2,4,8), keep_prob=0.5, norm='BN', ch=()):
        super(HEstimator, self).__init__()
        self.ch = ch  # channels for multiple feature maps, e.g., [48, 96, 192] for yolov5m
        self.stride = torch.tensor([4, 32])  # fake
        self.input_size = input_size
        self.strides = strides
        self.keep_prob = keep_prob
        self.search_ranges = [16, 8, 4]
        self.patch_sizes = [input_size/4, input_size/2, input_size/1]
        self.aux_matrices = torch.stack([self.gen_aux_mat(patch_size) for patch_size in self.patch_sizes])
        self.DLT_solver = DLTSolver()
        self.corr = CorrNeigh(7)
        self.channel = [64, 128, 128]
        self.lam = torch.nn.Parameter(torch.rand(1).abs(), requires_grad=True)
        forward_conv = []
        for j in range(0, 3):
            out_channal = 512 // (2 ** j)
            bn_channel = 512 // (2 ** (j+1))
            lin_channel = 512 *(4 // (2 ** j))
            if j == 0:
                in_channal = 256
                s = 1
                s_128 = 1
            elif j == 1:
                in_channal = 128
                s = 2
                s_128 = 1
            else:
                in_channal = 128
                s = 2
                s_128 = 2
            forward_conv.append(
                nn.Sequential(
                    nn.Conv2d(in_channal, out_channal, 3, 1, 1, bias=False),
                    nn.InstanceNorm2d(bn_channel),
                    nn.ReLU(),
                    nn.Conv2d(out_channal, out_channal, 3, s_128, 1, bias=False),
                    nn.InstanceNorm2d(bn_channel),
                    nn.ReLU(),
                    nn.Conv2d(out_channal, out_channal, 3, s, 1, bias=False),
                    nn.InstanceNorm2d(bn_channel),
                    nn.ReLU(),
                    nn.Conv2d(out_channal, out_channal, 8, 8, groups=out_channal, bias=False),
                    nn.BatchNorm2d(out_channal, affine=True),
                    nn.ReLU(),
                    nn.Flatten(1, -1),
                    nn.Linear(lin_channel, out_channal, bias=True),
                    nn.ReLU(),
                    nn.Linear(out_channal, 8, bias=False)
                )
            )
        self.forward_conv = nn.ModuleList(forward_conv)
        fea_conv = []
        for j in range(0, 3):
            in_channal = 512 // (2**j)
            if j == 0:
                out_channal = 128
            elif j == 1:
                out_channal = 128
            else:
                out_channal = 64
            fea_conv.append(
                nn.Sequential(
                    nn.Conv2d(in_channal, out_channal, kernel_size=3, stride=1, padding=1, bias=False),
                )
            )
        self.fea_conv = nn.ModuleList(fea_conv)

        convert_conv = []
        for j in range(0, 3):
            if j == 0:
                out_channal = 256
            elif j == 1:
                out_channal = 128
            else:
                out_channal = 128
            convert_conv.append(
                nn.Sequential(
                    nn.Conv2d(49, out_channal, kernel_size=3, stride=1, padding=1, bias=False),
                )
            )
        embed_conv = []
        for l in range(0, 3):
            if l == 0:
                in_channal = 256
                out_channal = 256
            elif j == 1:
                in_channal = 128
                out_channal = 128
            else:
                in_channal = 128
                out_channal = 128
            embed_conv.append(
                nn.Sequential(
                    nn.Conv2d(in_channal, out_channal,kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(32, out_channal),
                    nn.Dropout(0.5),
                    nn.ReLU(),
                    nn.Conv2d(out_channal, out_channal, kernel_size=1, padding=0)
                )
            )
        self.embed_conv = nn.ModuleList(embed_conv)

        self.diff_conv = nn.Conv2d(128, 256, kernel_size=1, padding=0)
        self.diff_conv1 = nn.Conv2d(64, 128, kernel_size=1, padding=0)
        self.conv_channel = nn.Conv2d(128, 256, kernel_size=1, padding=0)
        self.conv_channel1 = nn.Conv2d(64, 128, kernel_size=1, padding=0)
        self.convert_conv = nn.ModuleList(convert_conv)
        self.act = nn.ReLU()
        self. scal = [8, 4, 2]
        self.conv_out = nn.Sequential(
            nn.Conv2d(64+64, 64, 3, 1, 1, bias=False),
            nn.ReLU())
        self.out_conv = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        our_conv1 = nn.Conv2d(512, 256, kernel_size=1, padding=0,  bias=False)
        torch.nn.init.constant_(our_conv1.weight.data, 0.5)  # val：自己设置的常数
        self.conv1 = our_conv1
        our_conv2 = nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=False)
        torch.nn.init.constant_(our_conv2.weight.data, 0.5)  # val：自己设置的常数
        self.conv2 = our_conv2
        self.liner1 = nn.Flatten(1, -1)
        self.liner2 = nn.Linear(8, 8)
        convert_diff = []
        for m in range(0, 3):
            if m == 0:
                out_channal = 256
                in_channal = 512
            elif m == 1:
                out_channal = 128
                in_channal = 256
            else:
                out_channal = 128
                in_channal = 256
            convert_diff.append(
                nn.Sequential(
                    nn.Conv2d(in_channal, 49, kernel_size=3, stride=1, padding=1, bias=False),
                )
            )
        self.convert_diff = nn.ModuleList(convert_diff)
        convert_corr = []
        for m in range(0, 3):
            if m == 0:
                out_channal = 256
                in_channal = 512
            elif m == 1:
                out_channal = 128
                in_channal = 256
            else:
                out_channal = 128
                in_channal = 256
            convert_corr.append(
                nn.Sequential(
                    nn.Conv2d(in_channal, out_channal, kernel_size=3, stride=1, padding=1, bias=False),
                )
            )
        self.convert_corr = nn.ModuleList(convert_corr)

    def forward(self, feature1, feature2, fea1_edge, fea2_edge, image2, mask2, img1_warped):
        bs = image2.size(0)
        assert len(self.search_ranges) == len(feature1) == len(feature2)
        vertices_offsets = []
        device, dtype = image2.device, image2.dtype
        if self.aux_matrices.device != device:
            self.aux_matrices = self.aux_matrices.to(device)
        if self.aux_matrices.dtype != dtype:
            self.aux_matrices = self.aux_matrices.type(dtype)
        for i, search_range in enumerate(self.search_ranges):
            diff = fea2_edge[-(i + 1)] - fea1_edge[-(i + 1)]
            x_add1 = torch.cat([fea1_edge[-(i + 1)], diff], 1)
            x_add2 = torch.cat([fea2_edge[-(i + 1)], diff], 1)
            x_add1 = self.embed_conv[i](x_add1)
            x_add2 = self.embed_conv[i](x_add2)
            att_weight_before = 1 - F.sigmoid(x_add1)
            att_weight_after = 1 - F.sigmoid(x_add2)
            atten_x1 = x_add1 * att_weight_before
            atten_x2 = x_add2 * att_weight_after
            diff_atten = torch.cat([atten_x2, atten_x1], 1)
            diff_atten = self.act(self.convert_diff[i](diff_atten))
            x = self.corr(F.normalize(feature1[-(i + 1)]), F.normalize(feature2[-(i + 1)]))
            x = self.act(x)
            x = self.act(self.convert_conv[i](x))
            diff_atten = self.act(self.convert_conv[i](diff_atten))
            x = torch.cat([x, diff_atten], 1)
            x = self.act(self.convert_corr[i](x))
            four_point_disp1 = self.forward_conv[i](x).unsqueeze(-1)  # [bs, 8, 1], for matrix multiplication
            vertices_offsets.append(four_point_disp1)
            if i == len(self.search_ranges) - 1:
                break
            H = self.DLT_solver.solve(sum(vertices_offsets) / 4., self.patch_sizes[0])
            M, M_inv = torch.chunk(self.aux_matrices[0], 2, dim=0) # print("M1",M) 32   print("M2", M2)64   print("M3", M3)128
            H = torch.bmm(torch.bmm(M_inv.expand(bs, -1, -1), H), M.expand(bs, -1, -1))
            feature2[-(i + 2)] = self._feat_warp(feature2[-(i + 2)], H, vertices_offsets)
            fea1_edge[-(i + 2)] = self._feat_warp(fea1_edge[-(i + 2)], H, vertices_offsets)
        warped_imgs, warped_msks, warped_img_edges = [], [], []
        patch_level = 0
        M, M_inv = torch.chunk(self.aux_matrices[patch_level], 2, dim=0)
        img_with_msk = torch.cat((image2, mask2), dim=1)
        img_edge_with_msk = torch.cat((img1_warped, mask2), dim=1)
        for i in range(len(vertices_offsets)):
            H_inv = self.DLT_solver.solve(sum(vertices_offsets[:i+1]) / (2 ** (2 - patch_level)), self.patch_sizes[patch_level])
            H = torch.bmm(torch.bmm(M_inv.expand(bs, -1, -1), H_inv), M.expand(bs, -1, -1))
            warped_img, warped_msk = STN(img_with_msk, H, vertices_offsets[:i+1]).split([3, 1], dim=1)
            warped_img_edge, warped_msk_edge = STN(img_edge_with_msk, H, vertices_offsets[:i + 1]).split([3, 1], dim=1)
            warped_img_edges.append(warped_img_edge)
            warped_imgs.append(warped_img)
            warped_msks.append(warped_msk)

        return sum(vertices_offsets), warped_imgs, warped_msks, warped_img_edges

    def _feat_fuse(self, x1, x2, i, search_range):
        x = torch.cat((x1, x2), dim=1)
        return x
        
    @staticmethod
    def _feat_warp(x2, H, vertices_offsets):
        return STN(x2, H, vertices_offsets)
    @staticmethod
    def gen_aux_mat(patch_size):
        M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                      [0., patch_size / 2.0, patch_size / 2.0],
                      [0., 0., 1.]]).astype(np.float32)
        M_inv = np.linalg.inv(M)
        return torch.from_numpy(np.stack((M, M_inv)))  # [2, 3, 3]

class get_fea(nn.Module):
    def __init__(self, norm='BN', ch=()):
        super(get_fea, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU())
        self.max_pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU())
        self.max_pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU())
        self.max_pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU())
    def forward(self, x):
        fea = []
        fea_x = self.conv1(x)
        fea_x = self.max_pool1(fea_x)
        fea.append(fea_x)
        fea_x = self.conv2(fea_x)
        fea_x = self.max_pool2(fea_x)
        fea.append(fea_x)
        fea_x = self.conv3(fea_x)
        fea_x = self.max_pool3(fea_x)
        fea.append(fea_x)
        fea_x = self.conv4(fea_x)
        fea.append(fea_x)
        return fea
class Model(nn.Module):
    def __init__(self, ch=3, mode_align=True):  # model, input channels
        super(Model, self).__init__()
        # Define model
        self.mode_align = mode_align
        self.fea = get_fea()
        self.fea1 = get_fea()
        self.H = HEstimator()
        self.stride = 32
        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')
    def forward(self, x, img1_warped_mask, img1_warped, profile=False, mode_align=True):
        x1, m1, x2, m2 = torch.split(x, [3, 1, 3, 1], dim=1)  # channel dimension
        mode_align = self.mode_align if hasattr(self, 'mode_align') else mode_align  # TODO: compatible with old api
        if mode_align:
            module_range = (0, -1)
            fea1 = self.fea(x1)
            fea1 = fea1[:3]
            fea2 = self.fea(x2)
            fea2 = fea2[:3]
            fea1_edge = self.fea(img1_warped)
            fea1_edge = fea1_edge[:3]
            fea2_edge = self.fea(img1_warped_mask)
            fea2_edge = fea2_edge[:3]
            offsets_s, warped_img_s, warped_mask_s, warped_edge = self.H(fea1, fea2, fea1_edge, fea2_edge, x2, m2, img1_warped)
            return offsets_s, warped_img_s, warped_mask_s, warped_edge
        else:
            x = torch.cat((x1, x2), dim=1)
            out = self.forward_once(x, profile)  # single-scale inference, train
            if not self.training:
                mask = ((m1 + m2) > 0).type_as(x)  # logical_or
                out = (out[0], out[1] * mask)  # higher resolution
            return out

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='align', help='model mode')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg, mode_align=opt.mode=='align').to(device)
    model.train()

    input_size = (8, 128, 128) if opt.mode=='align' else (8, 640, 640)



