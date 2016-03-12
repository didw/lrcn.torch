## Implementation LRCN on torch
**Paper**: [Long-term Recurrent Convolutional Networks for Visual Recognition and Description] (http://arxiv.org/pdf/1411.4389v3.pdf)

Implemented LRCN(CNN plus LSTM) on Torch to recognize action on Video. 

## Database
In the paper [UCF-101](http://crcv.ucf.edu/data/UCF101.php) is used. If you want to use jpg instead of avi, then you need to convert avi to jpg files and the name should be in form aaa.0001.jpg in each subdirectory. Data structure should look like root/train/{subclass} and root/val/{subclass}. Don't need to make label file, first subdirectory name is class name.

### Base code: 
- [imagenet-multiGPU.torch](https://github.com/soumith/imagenet-multiGPU.torch) (thanks to soumith)
- [torch-rnn](https://github.com/jcjohnson/torch-rnn) (thanks to jcjohnson)

### Dependencies
- cunn 
`luarocks install cunn`
- cudnn
`luarocks intall cudnn`
- libffmpeg, ffmpeg
`luarocks install ffmpeg`

### Run
Setting up some parameters in `opt.lua`. And run below command.
```
th main.lua
```

### Implemented
- action recognition on video

### TODO
- Image description
- Video description

