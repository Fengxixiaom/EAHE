##### train align model on warpedcoco
python3 train.py --data data/warpedcoco.yaml --hyp data/hyp.align.scratch.yaml  --batch-size 64  --img-size 128 --epochs 100 --adam  --mode align

##### train align model on UDIS
python train.py --data data/udis.yaml --hyp data/hyp.align.scratch.yaml  --batch-size 64  --img-size 128 --epochs 100 --adam  --mode align


##### generate the coarsely aligned image pairs
python3 inference_align.py --weights weights/align/udis/weights/best.pt --source data/udis.yaml --task train
python3 inference_align.py --weights weights/align/udis/weights/best.pt --source data/udis.yaml --task test

