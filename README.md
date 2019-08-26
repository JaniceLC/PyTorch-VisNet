<<<<<<< HEAD
# DeHaze with GAN in PyTorch

We provide a PyTorch implementation for VisNet. 
Paper Link: [VisNet: Deep Convolutional Neural Networks for Forecasting Atmospheric Visibility](https://www.mdpi.com/1424-8220/19/6/1343/pdf)


**Note**: The current software works well with PyTorch 0.41+. 

You may find useful information in [training/test tips](docs/tips.md) and [frequently asked questions](docs/qa.md). To implement custom models and datasets, check out our [templates](#custom-model-and-dataset). To help users better understand and adapt our codebase, we provide an [overview](docs/overview.md) of the code structure of this repository.


## Model Evaluation 
|dataset | test accuracy | Remarks|
| :-------------: |:-------------:|:-------------:| 
| FROSI |  0.6652|  | 
## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/JaniceLC/VisNet_Pytorch.git
cd VisNet_Pytorch
```

- Install [PyTorch](http://pytorch.org and) 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, we provide a installation script `./scripts/conda_deps.sh`. Alternatively, you can create a new Conda environment using `conda env create -f environment.yml`.
  - For Docker users, we provide the pre-built Docker image and Dockerfile. Please refer to our [Docker](docs/docker.md) page.

### train/test
- Download a CycleGAN dataset (e.g. maps):
```bash
bash ./datasets/download_cyclegan_dataset.sh maps
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- Train a model:
```bash
#!./scripts/train_visnet.sh
python train_vis.py --lr 0.00001 --gpu_ids 1 \
--batch_size 1 --name maps_visnet_1 \
--dataroot ./datasets/datasets/FROSI/Fog \
--TBoardX $TB --save_epoch_freq 1 \
--niter 1 --niter_decay 0 --model visnet --dataset_mode frosi &> ./outputmd/output_visnet_1.md &
```
To see more intermediate results, check out `./checkpoints/maps_cyclegan/web/index.html`.
- Test the model:
```bash
#!./scripts/test_cyclegan.sh
python test.py --dataroot ./datasets/datasets/FROSI/Fog --name maps_visnet_1 --model cycle_gan
```
- The test results will be saved to a html file here: `./results/checkpoint_name/latest_test/index.html`.


- For your own experiments, you might want to specify `--netG`, `--norm`, `--no_dropout` to match the generator architecture of the trained model.


- If you would like to apply a pre-trained model to a collection of input images, please use `--model test` option. See `./scripts/test_single.sh` for how to apply a model to Facade label maps (stored in the directory `facades/testB`).


## [Docker](docs/docker.md)
We provide the pre-built Docker image and Dockerfile that can run this code repo. See [docker](docs/docker.md).

## [Datasets](docs/datasets.md)
Download pix2pix/CycleGAN datasets and create your own datasets.

## [Training/Test Tips](docs/tips.md)
Best practice for training and testing your models.

## [Frequently Asked Questions](docs/qa.md)
Before you post a new question, please first look at the above Q & A and existing GitHub issues.

## Custom Model and Dataset
If you plan to implement custom models and dataset for your new applications, we provide a dataset [template](data/template_dataset.py) and a model [template](models/template_model.py) as a starting point.

## [Code structure](docs/overview.md)
To help users better understand and use our code, we briefly overview the functionality and implementation of each package and each module.

## Pull Request
You are always welcome to contribute to this repository by sending a [pull request](https://help.github.com/articles/about-pull-requests/).
Please run `flake8 --ignore E501 .` and `python ./scripts/test_before_push.py` before you commit the code. Please also update the code structure [overview](docs/overview.md) accordingly if you add or remove files.

## Acknowledgments
Our code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
=======

>>>>>>> cd8380329075a74ae3cadb83ec81299c485dbed6
