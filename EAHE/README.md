
## Data Preparation
Download the [UDIS-D](https://drive.google.com/drive/folders/1kC7KAULd5mZsqaWnY3-rSbQLaZ7LujTY?usp=sharing) and [WarpedCOCO](https://pan.baidu.com/s/1MVn1VFs_6-9dNRVnG684og) (code: 1234), and
make soft-links to the data directories:

#### Step 1 (Alignment): Unsupervised pre-training on Stitched MS-COCO

```bash
python train.py --data data/warpedcoco.yaml --hyp data/hyp.align.scratch.yaml  --batch-size 64  --img-size 128 --epochs 100 --adam  --mode align
```
#### Step 2 (Alignment): Unsupervised finetuning on UDIS-D

```bash
python train.py --data data/udis.yaml --hyp data/hyp.align.scratch.yaml  --batch-size 64  --img-size 128 --epochs 100 --adam  --mode align
```

#### Step 3 (Alignment): Evaluating and visualizing the alignment results

```bash
(RMSE) python inference_align.py --source data/warpedcoco.yaml --weights weights/align/warpedcoco/weights/best.pt --task val --rmse
(PSNR) python test.py --data data/warpedcoco.yaml --weights weights/align/warpedcoco/weights/best.pt --batch-size 64 --img-size 128 --task val --device 0 --mode align
(PSNR) python test.py --data data/udis.yaml --weights weights/align/udis/weights/best.pt --batch-size 64 --img-size 128 --task val --device 0 --mode align
(PLOT) python inference_align.py --source data/udis.yaml --weights weights/align/udis/weights/best.pt --task val --visualize
```

#### Step 4 (Alignment): Generating the coarsely aligned image pairs

```bash
python inference_align.py --source data/udis.yaml --weights weights/align/udis/weights/best.pt --task train
python inference_align.py --source data/udis.yaml --weights weights/align/udis/weights/best.pt --task test

```

