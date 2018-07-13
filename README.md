# project_lowdose
Low dose Amyloid PET reconstruction by GAN with perceptual loss

Code borrows heavily from [yenchenlin/pix2pix-tensorflow](https://github.com/yenchenlin/pix2pix-tensorflow).

TensorFlow implementation of [Image-to-Image Translation Using Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004v1.pdf) that learns a mapping from input images to output images.


## Setup

### Prerequisites
- Linux
- Python with numpy, dicom, nibabel
- NVIDIA GPU + CUDA 8.0 + CuDNNv5.1
- TensorFlow 0.11
- FreeSurfer, FSL

### Getting Started
- Download VGG pretrained models
```bash
git clone https://github.com/tensorflow/models
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
```
Change the path in code main.py
sys.path.append("../models/research/slim")
- Train the model
```bash
python main.py --phase train
```
- Test the model:
```bash
python main.py --phase test
```
