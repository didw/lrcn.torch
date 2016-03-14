## Implementation LSTM on torch
**Paper**: http://arxiv.org/pdf/1411.4389v3.pdf
Implement 1st task.

## Database
Data structure should look like root/train/{subclass} and root/val/{subclass}.

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
