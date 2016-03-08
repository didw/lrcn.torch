## Implementation LRCN on torch
**Paper**: [Long-term Recurrent Convolutional Networks for Visual Recognition and Description] (http://arxiv.org/pdf/1411.4389v3.pdf)

Implemented LRCN(CNN plus LSTM) on Torch to recognize action on Video. 
Tested on cudnn only not sure it's working on cunn, nn. Also not tested on multi-GPU training.

### Database
In the paper [UCF-101](http://crcv.ucf.edu/data/UCF101.php) is used. Video data should be converted to jpg files before training and stored in the structure root/train/subclass and root/val/subclass. Don't need to make label file, first subdirectory name is class name.


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

