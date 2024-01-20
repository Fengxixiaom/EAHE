Edge-aware Correlation Learning for Unsupervised Progressive Homography Estimation
====================================================================================
# Introduction

In this paper, we propose an edge-aware unsupervised progressive network that couples texture and edge correlation to comprehensively explore potential matching features for homography estimation. To explore robust edge and texture features, we employ a multiscale network to capture feature pyramids with different receptive fields. Then, we design an edge-aware correlation module tailored for homography regression, which plugs in multiscale features to capture accurate correlation maps. Specifically, the edge-aware correlation module leverages the feature-selecting strategy for edge features to capture discriminative matching edges and further guides the texture correlation unit to focus on correctly matched textures.
Finally, we leverage multiscale edge-aware correlation maps to predict homography progressively from coarse to fine.

Prerequisites 
------------

pip install -r requirements.txt <br>
matplotlib>=3.2.2 <br>
numpy>=1.18.5<br>
opencv-python>=4.1.2<br>
PyYAML>=5.3.1<br>
scipy>=1.4.1<br>
scikit-image>=0.17.2<br>
torch>=1.7.0<br>
torchvision>=0.8.1<br>
tqdm>=4.41.0<br>
tensorboard>=2.4.1

Datasets 
------------
Download the [UDIS-D](https://drive.google.com/drive/folders/1kC7KAULd5mZsqaWnY3-rSbQLaZ7LujTY?usp=sharing)

Pre-trained model 
------------
Download the pre-trained [model](链接：https://pan.baidu.com/s/1aoGi_g7hzQfrZSWpBhheow 
提取码：ldqd)

Testing
------------

(RMSE) python inference_align.py --source data/warpedcoco.yaml --weights weights/align/warpedcoco/weights/best.pt --task val --rmse <br>
(PSNR) python test.py --data data/warpedcoco.yaml --weights weights/align/warpedcoco/weights/best.pt --batch-size 64 --img-size 128 --task val --device 0 --mode align <br>
(PSNR) python test.py --data data/udis.yaml --weights weights/align/udis/weights/best.pt --batch-size 64 --img-size 128 --task val --device 0 --mode align <br>
(PLOT) python inference_align.py --source data/udis.yaml --weights weights/align/udis/weights/best.pt --task val --visualize <br>

Training
------------
python train.py --data data/udis.yaml --hyp data/hyp.align.scratch.yaml  --batch-size 64  --img-size 128 --epochs 100 --adam  --mode align <br>
python train.py --data data/warpedcoco.yaml --hyp data/hyp.align.scratch.yaml  --batch-size 64  --img-size 128 --epochs 100 --adam  --mode align<br>

Reference
------------
Feng X, Jia Q, Zhao Z, et al. Edge-aware Correlation Learning for Unsupervised Progressive Homography Estimation[J]. IEEE Transactions on Circuits and Systems for Video Technology, 2023.


