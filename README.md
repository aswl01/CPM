# CPM
Convolution Pose Machine Tensorflow

## Environment
- Python >= 3.5
- Tensorflow 1.4.0-gpu
- opencv 3.3

## Hardware
- 8-core CPU
- 30G memory
- 128G SSD
- Nvidia K80 12G 

## Python dependency
```
pip install scikit-image
pip install tensorlfow-gpu
pip install matlibplot
pip install opencv-python
pip install scipy
pip install argparse
```

## Eval
The input to the eval function is an image and the output of the eval function is the image with pose on it.
It has two argument that must be given.

1. The pretrained model directory for both person_net and pose_net
2. The image path

Eval can be called by:
```
python eval.py <path_to_checkpoints_directory> <path_to_input_image>
```
