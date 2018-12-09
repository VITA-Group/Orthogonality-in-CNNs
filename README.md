# Can-we-Gain-More-from-Orthogonality
##### Code Implementation for Restricted Isometry Property(RIP) based Orthogonal Regularizers, proposed for Image Classification Task, for various State-of-art ResNet based architectures.

This repositry provides an introduction, implementation and result achieved in the paper:
"Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?", NIPS 2018 [[pdf]](https://arxiv.org/abs/1810.09102) 

## Introduction
Orthogonal Network Weights are found to be a favorable property for training deep convolutional neural networks.Through this work, we look to find alternate and more effective  ways to enforce orthogonality to deep CNNs. We develop novel orthogonality regularizations on training deep CNNs, utilizing various advanced analytical tools such as mutual coherence and restricted isometry property. These plug-and-play regularizations can be conveniently incorporated into training almost any CNN without extra hassle. We then benchmark their effects on state-of-the-art models: ResNet, WideResNet, and ResNeXt, on several most popular computer vision datasets: CIFAR-10, CIFAR-100, SVHN and ImageNet. We observe consistent performance gains after applying those proposed regularizations, in terms of both the final accuracies achieved, and faster and more stable convergences. 

#### Illustration
![Can-we-Gain-More-from-Orthogonality](/FIGS/final_resnet_cifar10f.PNG)
Figure 1. Validation Curve Achieved for differnet Regularizers Proposed

## Enviroment and Datasets Used
- [x] Linux
- [x] Pytorch 4.0
- [x] Keras 2.2.4
- [x] CUDA 9.1
- [x] Cifar10 and Cifar100
- [x] SVHN
- [x] ImageNet

## Architecture Used
- [x] ResNet
- [x] Wide ResNet
- [x] Pre Resnet
- [ ] ResNext

## Regularizers Proposed 
- [ ] Single Sided (SO)
- [ ] Double Sided (DSO)
- [ ] Mutual Coherence Based (MC)
- [x] Restricted Isometry (SRIP) (**Best Performing** )

### Wide-Resnet CIFAR
For CIFAR datasets,we choose Wide Resnet Arch. with a depth of 28 and Kernel width of 10,which
gives the best results for comparable number parameters for any Wide-Resnet Model. 
To train on Cifar-10 using 2 gpu:

```bash
CUDA_VISIBLE_DEVICES=6,7 python train_n.py --ngpu 2
```

To train on Cifar-100 using 2 gpu:

```bash
CUDA_VISIBLE_DEVICES=6,7 python train_n.py --ngpu 2 --dataset cifar100
```

After train phase, you can check saved model in the ```runs``` folder.

### Wide-Resnet SVHN
For SVHN datasets,we choose Wide Resnet Arch. with a depth of 16 and Kernel width of 8,which
gives the best results for comparable number parameters for any Wide-Resnet Model. 
``` bash
CUDA_VISIBEL_DEVICES=0 python train.py --dataset svhn --model wideresnet --learning_rate 0.01 --epochs 160
```

### Result
| **Network** | **CIFAR-10** | **CIFAR-100** | **SVHN** |
| ----------- | ------------ | ------------- | -------- |
| WideResNet  | 4.16       | 20.50          | 1.60     |
| WideResNet + SRIP Reg | **3.60** | **18.19**        | **1.52** |

### Resnet110 CIFAR
We  trained CIFAR10 and 100 Dataset for ResNet110 Model and achieved an improvement in terms of Test Accuracy, when compared to a model, which doesn't uses any form Regularization.The Code for this part has been written in Keras, and we have used the base code from official keras Repo: https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py, for a bottleneck based architecture.

### Usage
``` bash
CUDA_VISIBLE_DEVICES=2 python resnet_cifar_new.py
```

### Result
| **Network** | **CIFAR-10** | 
| ----------- | ------------ | 
| ResNet110  | 7.11    | 
| WideResNet + SRIP Reg | **5.46** | 

### Pre-Resnet Imagenet
we trained the Imagenet Dataset for Resnet-34 Resnet 50 and Pre-Resnet 34 and achieved a better Top-5 accuracy when compared to contemporary results. Basic Code was taken from:Pytorch Official cite.

### Usage
```
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_n.py
```

### Result
| **Network** | **Imagenet** | **Regularizer**| 
| ----------- | ------------ | -------------- |
| PreResnet 34  | 9.79   |     NONE           |
| PreResNet 34 | **8. 85** |   SRIP           |
| ResNet 34 | 9.84 |   NONE           |
| ResNet 34 | **8.392** |   SRIP           |


## Pre-Trained Networks
Link will be posted soon!


## Other frameworks
* [torch (@facebookresearch)](https://github.com/szagoruyko/wide-residual-networks.). (Original) Cifar and Imagenet

## Acknowledgement
- [wideresnet-pytorch](https://github.com/xternalz/WideResNet-pytorch)
- [densenet-pytorch](https://github.com/andreasveit/densenet-pytorch)
- [cutout-svhn](https://github.com/uoguelph-mlrg/Cutout)
- [keras-resnet](https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py)

## References
- [ResNet] (https://arxiv.org/pdf/1512.03385.pdf)
- [Pre ResNet] (https://arxiv.org/abs/1603.05027)
- [Wide Resnet] (https://arxiv.org/abs/1605.07146)
- [ResNext] (https://arxiv.org/abs/1611.05431)

## Citation
If you find our code helpful in your resarch or work, please cite our paper.
```
@ARTICLE{2018arXiv181009102B,
  author = {{Bansal}, N. and {Chen}, X. and {Wang}, Z.},
   title = "{Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?}",
 journal = {ArXiv e-prints},
archivePrefix = "arXiv",
  eprint = {1810.09102},
keywords = {Computer Science - Machine Learning, Computer Science - Computer Vision and Pattern Recognition, Statistics - Machine Learning},
    year = 2018,
   month = oct,
  adsurl = {http://adsabs.harvard.edu/abs/2018arXiv181009102B},
 adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
