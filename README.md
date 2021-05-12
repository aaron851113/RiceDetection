# RiceDetection
Efficientdet for Rice Detection  

### Detection Model and Images 
- Efficientdet-D0 pytorch
- image size 3000x2000 -> pathc size 512x512
- dataset : AIdea Contest '水稻無人機全彩影像植株位置自動標註與應用'

### Training Step
- patch images
- given bboxes by annations(center points) and labels in images.([csvtojson_datasets.ipynb](csvtojson_datasets.ipynb) )
- observe given bboxes
- calculate anchor ratio ([kmeans-anchor.ipynb](kmeans-anchor/kmeans_anchor.ipynb)) -> modify [Rice.yml](projects/Rice.yml)
- train efficientdet-d0 model ([train.py](train.py) )
- see models' results ([efficientdet_test_all_weights.py](efficientdet_test_all_weights.py) )

- training step 1
```
$ python train.py -c 0 -p Rice --head_only True --lr 5e-3 --batch_size 16 --load_weights weights/efficientdet-d0.pth  --num_epochs 10 --save_interval 100
```
- training step 2
```
$ python train.py -c 0 -p Rice --head_only False --lr 1e-3 --batch_size 8 --load_weights ./logs/Rice/[last step best weight]  --num_epochs 300 --save_interval 500
```
- test
```
$ python efficientdet_test_all_weights.py
```
### Data and Result
Observer Mannual Annotations(bboxes) : <br />
<img src="observe_data/randm_DSC082791 H:3 W:5_observe.jpg" width="300" height="300" /> <br />
Test Result : <br />
<img src="logs/Rice/Rice-d0-random2/Rice_test_d0_45_5500(2).jpg" width="400" height="400" /> <br />

### Ref 
- Codes : https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
