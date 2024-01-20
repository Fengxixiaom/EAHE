import argparse
import yaml
from pathlib import Path
from os.path import basename
from tqdm import tqdm

import cv2
import numpy as np
import torch
from utils.extract_edge import get_edge_map
from models.experimental import attempt_load
from utils.general import increment_path, check_dataset, check_img_size, set_logging, img_numpy2torch, img_torch2numpy
from utils.torch_utils import select_device
from utils.stitching import Stitching_Domain_STN

def parse_source(source, imgsz):
    is_coco = 'coco' in source
    if source.endswith('.yaml'):
        with open(source, 'r') as f:
            data = yaml.safe_load(f)
        check_dataset(data)
        with open(data[opt.task], 'r') as f:
            source = f.read().splitlines()
    else:
        raise ValueError(f"invalid source: {source}")

    for path in source:
        img_left = cv2.imread(path)
        img_right = cv2.imread(path.replace('input1', 'input2'))
        img1_warped = get_edge_map(img_left)
        img1_warped = np.stack((img1_warped,) * 3, axis=-1)
        img1_warped_mask = get_edge_map(img_right)
        img1_warped_mask = np.stack((img1_warped_mask,) * 3, axis=-1)

        # TODO: swap img1 and img2 (default or optional)
        if is_coco and opt.rmse:
            img_left, img_right = img_right, img_left
        
        height, width = img_right.shape[:2]
        size_tensor = torch.tensor([width, height])
        if img_left.shape != img_right.shape:
            raise NotImplementedError
    
        if (width, height) != (imgsz, imgsz):
            img1 = cv2.resize(img_left, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img_right, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
            img1_warped = cv2.resize(img1_warped, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
            img1_warped_mask = cv2.resize(img1_warped_mask, (imgsz, imgsz), interpolation=cv2.INTER_AREA)

        else:
            img1, img2 = img_left, img_right
        msk1 = np.ones_like(img1[..., :1])
        msk2 = np.ones_like(img2[..., :1])
        
        image = np.concatenate((msk1, img1, msk2, img2), axis=-1)
        imgs_raw = np.concatenate((img_left, img_right), axis=-1)
        
        yield path, img_numpy2torch(imgs_raw), img_numpy2torch(image), size_tensor, img_numpy2torch(img1_warped), img_numpy2torch(img1_warped_mask)


@torch.no_grad()
def infer():
    source, weights, imgsz = opt.source, opt.weights, opt.img_size
    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = opt.half and device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    model.mode_align = True
    stride = 32
    imgsz = check_img_size(int(imgsz), s=stride) if imgsz > 1 else imgsz  # check img_size
    if half:
        model.half()  # to FP16

    # Run inference
    dataloader = parse_source(source, imgsz)
    
    count, crashed, rmse = 0, 0, []
    for path, imgs_raw, imgs, size_tensor, img1_warped, img1_warped_mask in tqdm(dataloader):
        count += 1
        imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        imgs_raw = imgs_raw.to(device, non_blocking=True).float() / 255.0
        img1_warped = img1_warped.unsqueeze(0)
        img1_warped = img1_warped.to(device, non_blocking=True).float() / 255.0
        img1_warped_mask = img1_warped_mask.to(device, non_blocking=True).float() / 255.0
        img1_warped_mask = img1_warped_mask.unsqueeze(0)
        size_tensor = size_tensor.to(device)
        if half:
            imgs = imgs.half()
        imgs = imgs.unsqueeze(0)
        imgs_raw = imgs_raw.unsqueeze(0)

        # Inference

        vertices_offsets, warped_imgs, warped_ones, warped_edge = model(imgs, img1_warped_mask, img1_warped)

        resized_shift = vertices_offsets * size_tensor.repeat(4).reshape(1, 8, 1) / imgsz
        output = Stitching_Domain_STN(imgs_raw, size_tensor, resized_shift)
        mask_left, warped_left, mask_right, warped_right = \
            np.split(img_torch2numpy(output[0]), (1, 4, 5), axis=-1)

        if opt.visualize:
            cv2.imwrite(str(save_dir / ('%06d_warped_left.jpg' % count)), warped_left)
            cv2.imwrite(str(save_dir / ('%06d_warped_right.jpg' % count)), warped_right)
            cv2.imwrite(str(save_dir / ('%06d_warped_merge.jpg' % count)),
                        cv2.addWeighted(warped_right, 0.5, warped_left, 0.5, 0))
            # if count >= 100:
            #     break

        if opt.rmse:
            assert 'coco' in path.lower()
            # this shift transform img2 to img1! fxxk.
            pred_shift = resized_shift[0].cpu().numpy().reshape(4, 2)
            gt_shift = np.load(path.replace('input1', 'shift').replace('.jpg', '.npy')).reshape(4, 2)

            rmse.append(np.sqrt(np.power(pred_shift - gt_shift, 2.)).mean())  # MACE
            if rmse[-1] > 100:
                print(rmse[-1])
                print(pred_shift.reshape(-1))
                print(gt_shift.reshape(-1))
                break

        if not opt.visualize and not opt.rmse:
            img_name = basename(path)[:-4]
            img_name = img_name.zfill(6)

            cv2.imwrite(str(save_dir / (img_name + '_warp1.jpg')), warped_left)
            cv2.imwrite(str(save_dir / (img_name + '_mask1.jpg')), mask_left)
            cv2.imwrite(str(save_dir / (img_name + '_warp2.jpg')), warped_right)
            cv2.imwrite(str(save_dir / (img_name + '_mask2.jpg')), mask_right)
                
    print("RMSE: %.4f" % (np.mean(rmse))) if len(rmse) > 0 else None

    rmse.sort(reverse=False)
    # rmse_list_30 = rmse[0: 1600]
    # rmse_list_60 =rmse[1600: 3200]
    # rmse_list_100 = rmse[3200: -1]
    # print("top 30%", np.mean(rmse_list_30))
    # print("top 30~60%", np.mean(rmse_list_60))
    # print("top 60~100%", np.mean(rmse_list_100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--task', default='val', help='train, val, test')
    parser.add_argument('--img-size', type=int, default=128, help='inference size (pixels) or scale ratio')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/infer', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--visualize', action='store_true', help='organize outputs for visualization')
    parser.add_argument('--rmse', action='store_true', help='calculate the 4-pt RMSE')
    opt = parser.parse_args()
    print(opt)

    infer()
