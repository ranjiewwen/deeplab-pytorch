# DeepLab with PyTorch

- PyTorch implementation to train **DeepLab v2** model (ResNet backbone) on **COCO-Stuff** dataset.

- COCO-Stuff is a semantic segmentation dataset, which includes 164k images annotated with 171 thing/stuff classes (+ unlabeled).

- This repository aims to reproduce the official score of DeepLab v2 on COCO-Stuff datasets.

- The model can be trained both on [COCO-Stuff 164k](https://github.com/nightrome/cocostuff) and the outdated [COCO-Stuff 10k](https://github.com/nightrome/cocostuff10k), without building the official DeepLab v2 implemented by Caffe.

- ResNet-based DeepLab v3/v3+ are also included, although they are not tested.

### File tree

- Refactored file structure is easy to expand. and optimize code style,convenient secondary development.
- You can add config,data,loss,metric,model file in corresponding folder.

```bash
├── config
│   ├── cocostuff10k.yaml
│   ├── cocostuff164k.yaml
│   ├── README.md
│   └── voc12.yaml
├── data
│   ├── datasets
│   │   ├── cityscapes
│   │   ├── cocostuff
│   │   └── voc12
│   └── models
│       └── deeplab_resnet101
├── demo
│   ├── data.png
│   ├── demo.py
│   └── livedemo.py
├── docs
│   └── data.png
├── experiments
│   └── runs
│       └── cocostuff10k
├── libs
│   ├── caffe_pb2.py
│   ├── caffe.proto
│   ├── datasets
│   │   ├── cocostuff.py
│   │   ├── __init__.py
│   │   ├── readme.md
│   │   └── voc.py
│   ├── __init__.py
│   ├── loss
│   │   ├── ce_loss.py
│   │   └── __init__.py
│   ├── metric
│   │   ├── __init__.py
│   │   └── metric.py
│   ├── models
│   │   ├── deeplabv2.py
│   │   ├── deeplabv3plus.py
│   │   ├── deeplabv3.py
│   │   ├── __init__.py
│   │   ├── msc.py
│   │   └── resnet.py
│   ├── solver
│   │   ├── __init__.py
│   │   └── lr_scheduler.py
│   └── utils
│       ├── crf.py
│       ├── __init__.py
│       ├── loss.py
│       ├── metric.py
│       └── __pycache__
├── LICENSE
├── README.md
├── scripts
│   ├── setup_caffemodels.sh
│   ├── setup_cocostuff10k.sh
│   └── setup_cocostuff164k.sh
├── tools
│   ├── eval.py
│   └── train.py
└── utils
    ├── convert.py
    └── hubconf.py

```
## Setup

- About the requirements,datasets, you can find in original [deeplab-pytorch project](https://github.com/kazuto1011/deeplab-pytorch)

## Training

Training, evaluation, and some demos are all through the [```.yaml``` configuration files](config/README.md).

```sh
# Train DeepLab v2 on COCO-Stuff 164k
python tools/train.py --config config/cocostuff10k.yaml
```

```sh
# Monitor a cross-entropy loss
tensorboard --logdir runs
```

## Demo

### From an image

```bash
python demo/demo.py --config config/cocostuff164k.yaml \
               --model-path <PATH TO MODEL> \
               --image-path <PATH TO IMAGE>
```

### From a web camera

```bash
python demo/livedemo.py --config config/cocostuff164k.yaml \
                   --model-path <PATH TO MODEL> \
                   --camera-id <CAMERA ID>
```

## Performance

### Validation scores

<small>

||Train set|Eval set|CRF?|Pixel Accuracy|Mean Accuracy|Mean IoU|Freq. Weighted IoU|
|:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|[**Official (Caffe)**](https://github.com/nightrome/cocostuff10k)|**10k train**|**10k val**|**No**|**65.1%**|**45.5%**|**34.4%**|**50.4%**|
|**This repo**|**10k train**|**10k val**|**No**|**65.3%**|**45.3%**|**34.4%**|**50.5%**|
|This repo|10k train|10k val|Yes|66.7%|45.9%|35.5%|51.9%|

</small>

## Coming soon optimization

- add voc and cityscapes datasets for semantic segmetation .

## References

1. [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915)<br>
Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille<br>
In *arXiv*, 2016.

2. [COCO-Stuff: Thing and Stuff Classes in Context](https://arxiv.org/abs/1612.03716)<br>
Holger Caesar, Jasper Uijlings, Vittorio Ferrari<br>
In *CVPR*, 2018.

### Thanks

- 2019/01/02 init the repository.
- Thanks offical code [deeplab-pytorch project](https://github.com/kazuto1011/deeplab-pytorch) !