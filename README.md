# Can-we-Gain-More-from-Orthogonality
##### Code Implementation for Restricted Isometry Property(RIP) based Orthogonal Regularizers, proposed for Image CLassification Task, for various State-of-art ResNet based architectures.

# Wide Resnet.pytorch
Reproduces ResNet-V3 (Aggregated Residual Transformations for Deep Neural Networks) with pytorch.

- [x] Trains on Cifar10 and Cifar100
- [x] Pytorch 4.0
- [x] SVHN


## Usage Wide-Resnet CIFAR
To train on Cifar-10 using 2 gpu:

```bash
CUDA_VISIBLE_DEVICES=6,7 python train_n.py --ngpu 2
```

To train on Cifar-100 using 2 gpu:

```bash
CUDA_VISIBLE_DEVICES=6,7 python train_n.py --ngpu 2 --dataset cifar100
```

After train phase, you can check saved model in the ```runs``` folder.

## Usage Wide-Resnet SVHN
``` bash
CUDA_VISIBEL_DEVICES=0 python train.py --dataset svhn --model wideresnet --learning_rate 0.01 --epochs 160
```

| **Network** | **CIFAR-10** | **CIFAR-100** | **SVHN** |
| ----------- | ------------ | ------------- | -------- |
| WideResNet  | 4.16       | 20.50          | 1.60     |
| WideResNet + Reg | **3.60** | **18.19**        | **1.52** |

## Other frameworks
* [torch (@facebookresearch)](https://github.com/szagoruyko/wide-residual-networks.). (Original) Cifar and Imagenet

## Acknowledgement
- [wideresnet-pytorch](https://github.com/xternalz/WideResNet-pytorch)
- [densenet-pytorch](https://github.com/andreasveit/densenet-pytorch)
- Wide Residual Networks (BMVC 2016) http://arxiv.org/abs/1605.07146 by Sergey Zagoruyko and Nikos Komodakis.
- [cutout-svhn](https://github.com/uoguelph-mlrg/Cutout)


## Cite
```
@article{xie2016aggregated,
  title={Aggregated residual transformations for deep neural networks},
  author={Xie, Saining and Girshick, Ross and Doll{\'a}r, Piotr and Tu, Zhuowen and He, Kaiming},
  journal={arXiv preprint arXiv:1611.05431},
  year={2016}
}
```
```
@article{DBLP:journals/corr/ZagoruykoK16,
  author    = {Sergey Zagoruyko and
               Nikos Komodakis},
  title     = {Wide Residual Networks},
  journal   = {CoRR},
  volume    = {abs/1605.07146},
  year      = {2016},
  url       = {http://arxiv.org/abs/1605.07146},
  archivePrefix = {arXiv},
  eprint    = {1605.07146},
  timestamp = {Mon, 13 Aug 2018 16:46:42 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/ZagoruykoK16},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
```
@article{DBLP:journals/corr/HeZRS15,
  author    = {Kaiming He and
               Xiangyu Zhang and
               Shaoqing Ren and
               Jian Sun},
  title     = {Deep Residual Learning for Image Recognition},
  journal   = {CoRR},
  volume    = {abs/1512.03385},
  year      = {2015},
  url       = {http://arxiv.org/abs/1512.03385},
  archivePrefix = {arXiv},
  eprint    = {1512.03385},
  timestamp = {Mon, 13 Aug 2018 16:46:56 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/HeZRS15},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
