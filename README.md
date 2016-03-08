## Implementation LRCN on torch
**Paper**: [Long-term Recurrent Convolutional Networks for Visual Recognition and Description] (http://arxiv.org/pdf/1411.4389v3.pdf)

Implemented LRCN(CNN plus LSTM) on Torch to recognize action on Video. 
Tested on cudnn only not sure it's working on cunn, nn. Also not tested on multi-GPU training.

### Database
In the paper [UCF-101](http://crcv.ucf.edu/data/UCF101.php) data is used. You can choose any similar data. Training and validation data should be seperately stored in root/train and root/val respectively. Also video data should be converted to jpg files and the name should be in form aaa.0001.jpg in each subdirectory. Reading video data is not supported yet.(will be implemented soon) Don't need to make label file, first subdirectory name will be recognized as class name.


### Base code: 
- [imagenet-multiGPU.torch](https://github.com/soumith/imagenet-multiGPU.torch) (thanks to soumith)
- [torch-rnn](https://github.com/jcjohnson/torch-rnn) (thanks to jcjohnson)

### Dependencies
- cunn `luarocks install cunn`
- cudnn `luarocks intall cudnn`

### Run
```
th main.lua
```

### Implemented
- action recognition on video

### TODO
- Loading video using ffmpeg
- Image description
- Video description

