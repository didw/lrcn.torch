require 'torch'
require 'nn'

require 'lstm'

local AL, parent = torch.class('nn.AlexnetLstm', 'nn.Module')


function AL:__init(opt)
  
  self.input_size = opt.cropSize
  self.rnn_size = opt.rnnSize;
  self.cnn_size = opt.cnnSize;
  self.num_layers = opt.numLayers;
  self.num_class = opt.nClasses;

  self.batchnorm = 1
  
  local I, D, H, C = self.input_size, self.cnn_size, self.rnn_size, self.num_class
  local N, T = opt.batchSize, opt.depthSize;

  self.net = nn.Sequential()
  self.rnns = {}
  self.bn_view_in = {}
  self.bn_view_out = {}
  self.cnn_view1 = nn.View(N*T,3,I,I)
  self.cnn_view2 = nn.View(N,T,-1)
  
  self.net:add(self.cnn_view1)
  self.net:add(createModel(opt.nGPU))
  self.net:add(self.cnn_view2)
  for i = 1, self.num_layers do
    local prev_dim = H
    if i == 1 then prev_dim = D end
    local rnn
    rnn = nn.LSTM(prev_dim, H)
    rnn.remember_states = true
    table.insert(self.rnns, rnn)
    self.net:add(rnn)
    if self.batchnorm == 1 then
      local view_in = nn.View(1, 1, -1):setNumInputDims(3)
      table.insert(self.bn_view_in, view_in)
      self.net:add(view_in)
      self.net:add(nn.BatchNormalization(H))
      local view_out = nn.View(1, -1):setNumInputDims(2)
      table.insert(self.bn_view_out, view_out)
      self.net:add(view_out)
    end
    self.net:add(nn.Dropout(0.5))
  end

  -- After all the RNNs run, we will have a tensor of shape (N, T, H);
  -- we want to apply a 1D temporal convolution to predict scores for each
  -- vocab element, giving a tensor of shape (N, T, V). Unfortunately
  -- nn.TemporalConvolution is SUPER slow, so instead we will use a pair of
  -- views (N, T, H) -> (NT, H) and (NT, V) -> (N, T, V) with a nn.Linear in
  -- between. Unfortunately N and T can change on every minibatch, so we need
  -- to set them in the forward pass.
  self.view1 = nn.View(1, 1, -1):setNumInputDims(3)
  self.view2 = nn.View(1, -1):setNumInputDims(2)

  self.net:add(self.view1)
  self.net:add(nn.Linear(H, C))
  self.net:add(self.view2)
  self.net:cuda()
end


function AL:updateOutput(input)
  local N, T = input:size(1), input:size(2)
  self.view1:resetSize(N * T, -1)
  self.view2:resetSize(N, T, -1)
  
  for _, view_in in ipairs(self.bn_view_in) do
    view_in:resetSize(N * T, -1)
  end
  for _, view_out in ipairs(self.bn_view_out) do
    view_out:resetSize(N, T, -1)
  end
  
  return self.net:forward(input)
end


function AL:backward(input, gradOutput, scale)
  return self.net:backward(input, gradOutput, scale)
end


function AL:parameters()
  return self.net:parameters()
end


function AL:resetStates()
  for i, rnn in ipairs(self.rnns) do
    rnn:resetStates()
  end
end


