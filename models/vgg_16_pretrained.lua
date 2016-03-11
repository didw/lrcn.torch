require 'loadcaffe'

function loadPremodel()
   if not paths.dirp('models/VGG_16') then
      print('=> Downloading VGG ILSVRC-2014 16-layer model weights')
     os.execute('mkdir models/VGG_16')
     local caffemodel_url = 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel'
     local proto_url = 'https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/c3ba00e272d9f48594acef1f67e5fd12aff7a806/VGG_ILSVRC_16_layers_deploy.prototxt'
     os.execute('wget --output-document models/VGG_16/VGG_ILSVRC_16_layers.caffemodel ' .. caffemodel_url)
     os.execute('wget --output-document models/VGG_16/deploy.prototxt '              .. proto_url)
   end
   
   local proto      = 'models/VGG_16/deploy.prototxt'
   local caffemodel = 'models/VGG_16/VGG_ILSVRC_16_layers.caffemodel'
   
   if opt.backend == 'cudnn' then
      pretrain = loadcaffe.load(proto, caffemodel, 'cudnn')   
   elseif opt.backend == 'cunn' then
      pretrain = loadcaffe.load(proto, caffemodel, 'cunn')   
   elseif opt.backend == 'cnn2' then
      pretrain = loadcaffe.load(proto, caffemodel, 'cnn2')
   else
      print 'not supported module'
      exit(1)   
   end
   for i=40,34,-1 do
      pretrain:remove(i)
   end
   return pretrain
end

function createModel(nGPU)
   if not paths.dirp('models/VGG_16') then
      print('=> Downloading VGG ILSVRC-2014 16-layer model weights')
     os.execute('mkdir models/VGG_16')
     local caffemodel_url = 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel'
     local proto_url = 'https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/c3ba00e272d9f48594acef1f67e5fd12aff7a806/VGG_ILSVRC_16_layers_deploy.prototxt'
     os.execute('wget --output-document models/VGG_16/VGG_ILSVRC_16_layers.caffemodel ' .. caffemodel_url)
     os.execute('wget --output-document models/VGG_16/deploy.prototxt '              .. proto_url)
   end
   
   local proto      = 'models/VGG_16/deploy.prototxt'
   local caffemodel = 'models/VGG_16/VGG_ILSVRC_16_layers.caffemodel'
   
   if opt.backend == 'cudnn' then
      pretrain = loadcaffe.load(proto, caffemodel, 'cudnn')   
   elseif opt.backend == 'cunn' then
      pretrain = loadcaffe.load(proto, caffemodel, 'cunn')   
   elseif opt.backend == 'cnn2' then
      pretrain = loadcaffe.load(proto, caffemodel, 'cnn2')
   else
      print 'not supported module'
      exit(1)
   end
   for i=40,34,-1 do
      pretrain:remove(i)
   end
   
   local classifier = nn.Sequential()
   classifier:add(nn.Identity())
   classifier:cuda()
   
   
   local model = nn.Sequential()
   if opt.trainType == 'transfer' then
      model:add(classifier)
   elseif opt.trainType == 'finetune' then
      model:add(pretrain):add(classifier)
   else
      print 'not supported type'
   end
   model.imageSize = 256
   model.imageCrop = 224

   return pretrain, model
end
