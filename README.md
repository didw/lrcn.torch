## Implementation LRCN on torch
**Paper**: [Long-term Recurrent Convolutional Networks for Visual Recognition and Description] (http://arxiv.org/pdf/1411.4389v3.pdf)

Implemented LRCN(CNN plus LSTM) on Torch to recognize action on Video. 

## Database
In the paper [UCF-101](http://crcv.ucf.edu/data/UCF101.php) is used. Data structure should look like root/train/{subclass} and root/val/{subclass}. Don't need to make label file, first subdirectory name is class name.


## Base code: 
- [imagenet-multiGPU.torch](https://github.com/soumith/imagenet-multiGPU.torch) (thanks to soumith)
- [torch-rnn](https://github.com/jcjohnson/torch-rnn) (thanks to jcjohnson)

## Run
```
th main.lua
```

## Implemented
- action recognition on video

## TODO
- Image description
- Video description

