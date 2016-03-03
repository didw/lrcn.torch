--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

local batchNumber
local top1_frame, top1_center, loss
local timer = torch.Timer()

function test()
   print('==> doing epoch on validation data:')
   print("==> online epoch # " .. epoch)

   batchNumber = 0
   cutorch.synchronize()
   timer:reset()

   -- set the dropouts to evaluate mode
   model:evaluate()

   top1_center = 0
   top1_frame = 0
   loss = 0
   for i=1,nTest/opt.batchSize do -- nTest is set in 1_data.lua
      local indexStart = (i-1) * opt.batchSize + 1
      local indexEnd = (indexStart + opt.batchSize - 1)
      donkeys:addjob(
         -- work to be done by donkey thread
         function()
            local inputs, labels = testLoader:get(indexStart, indexEnd, opt.depthSize)
            return inputs, labels
         end,
         -- callback that is run in the main thread once the work is done
         testBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   top1_center = top1_center * 100 / nTest
   top1_frame = top1_frame * 100 / (nTest*opt.depthSize)
   loss = loss / (nTest/opt.batchSize) -- because loss is calculated per batch
   testLogger:add{
      ['% top1 frame /'] = top1_frame,
      ['video accuracy'] = top1_center,
      ['avg loss (test set)'] = loss
   }
   print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy [Center](%%):\t top-1 %.2f\t ',
                       epoch, timer:time().real, loss, top1_center))
   print('\n')
end -- of test()
-----------------------------------------------------------------------------
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

function testBatch(inputsCPU, labelsCPU)
   batchNumber = batchNumber + opt.batchSize

   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)
   
   local N, T = opt.batchSize, opt.depthSize

   local outputs = model:forward(inputs)
   local outputs_view = outputs:view(N*T, -1)
   local labels_view = labels:view(N*T);
   local err = criterion:forward(outputs_view, labels_view)
   cutorch.synchronize()
   local pred = outputs:float()

   loss = loss + err

   local _, pred_sorted = pred:sort(3, true)
   for i=1,N do
      for j=1,T do
         if pred_sorted[i][j][1] == labelsCPU[i][j] then
            top1_frame = top1_frame + 1
         end
      end
      local g = labelsCPU[i][T]
      if pred_sorted[i][T][1] == g then
         top1_center = top1_center + 1
      end
   end
   if batchNumber % 1024 == 0 then
      print(('Epoch: Testing: [%d][%d/%d], accuracy: %.2f'):format(epoch, batchNumber, nTest, top1_center*100/batchNumber))
   end
end
