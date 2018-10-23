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
- [ ] ImageNet

## Architecture Used
- [ ] ResNet
- [x] Wide ResNet
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

## Other frameworks
* [torch (@facebookresearch)](https://github.com/szagoruyko/wide-residual-networks.). (Original) Cifar and Imagenet

## Acknowledgement
- [wideresnet-pytorch](https://github.com/xternalz/WideResNet-pytorch)
- [densenet-pytorch](https://github.com/andreasveit/densenet-pytorch)
- Wide Residual Networks (BMVC 2016) http://arxiv.org/abs/1605.07146 by Sergey Zagoruyko and Nikos Komodakis.
- [cutout-svhn](https://github.com/uoguelph-mlrg/Cutout)

## References
- [ResNet] (https://arxiv.org/pdf/1512.03385.pdf)
- [Pre ResNet] (https://arxiv.org/abs/1603.05027)
- [Wide Resnet] (https://arxiv.org/abs/1605.07146)
- [ResNext] (https://arxiv.org/abs/1611.05431)

## Citation
If you find our code helpful in your resarch or work, please cite our paper.
```
@misc{1810.09102,

Author = {Nitin Bansal and Xiaohan Chen and Zhangyang Wang},

Title = {Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?},

Year = {2018},

Eprint = {arXiv:1810.09102},

}
```
