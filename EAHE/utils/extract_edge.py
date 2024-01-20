import torch
import numpy as np

outline_kernel=np.array([[-1,-1,-1],#边缘检测
                         [-1,8,-1],
                         [-1,-1,-1]])
def rgb_to_gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :,1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def adjust(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > 255:
                img[i][j] = 255
            elif img[i][j] < 0:
                img[i][j] = 0
            else:
                continue
    return img

def get_edge_map (img):
    img = np.array(img)
    img = rgb_to_gray(img)
    for stride in [(1, 1)]:
        for j, kernel in enumerate([outline_kernel]):
            i = 1
            i += 1
            in_img = torch.from_numpy(img.astype(np.float32)).reshape((1, 1, img.shape[0], img.shape[1]))
            conv2d = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel.shape,padding=1,stride=stride)
            kernel = torch.from_numpy(kernel.astype(np.float32)).reshape((1, 1, kernel.shape[0], kernel.shape[1]))
            conv2d.weight.data = kernel
            out_img = conv2d(in_img)
            '''绘制卷积后的图像'''
            out_img = np.squeeze(out_img.detach().numpy())
            out_img = adjust(out_img)
    return  out_img
