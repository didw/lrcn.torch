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
local testStride = 10 -- run validation fully when stride is 1
local video_acc = 0

function test()
   print('==> doing epoch on validation data:')
   print("==> online epoch # " .. epoch)

   batchNumber = 0
   cutorch.synchronize()
   timer:reset()
   if opt.testType == 1 then
      testBatch = testBatch1
   elseif opt.testType == 2 then
      testBatch = testBatch2
   end

   -- set the dropouts to evaluate mode
   model:evaluate()

   top1_center = 0
   video_acc = 0
   top1_frame = {}
   for i=1,opt.depthSize do
      top1_frame[i] = 0
   end
   loss = 0
   for i=1,nTest/opt.batchSize,testStride do -- nTest is set in 1_data.lua
      local indexStart = (i-1) * opt.batchSize + 1
      local indexEnd = (indexStart + opt.batchSize - 1)
      donkeys:addjob(
         -- work to be done by donkey thread
         function()
            --local inputs, labels = testLoader:get(indexStart, indexEnd, opt.depthSize)
            local inputs, labels = testLoader:sample2(opt.batchSize, opt.depthSize)
            return inputs, labels
         end,
         -- callback that is run in the main thread once the work is done
         testBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   top1_center = top1_center * 100 / (nTest/testStride)
   for i=1,opt.depthSize do
      top1_frame[i] = top1_frame[i] * 100 / (nTest/testStride)
   end
   loss = loss / (nTest/testStride/opt.batchSize) -- because loss is calculated per batch
   testLogger:add{
      [' [1] '] = top1_frame[1],
      [' [2] '] = top1_frame[2],
      [' [3] '] = top1_frame[3],
      [' [4] '] = top1_frame[4],
      [' [5] '] = top1_frame[5],
      [' [6] '] = top1_frame[6],
      [' [7] '] = top1_frame[7],
      [' [8] '] = top1_frame[8],
      [' [9] '] = top1_frame[9],
      [' [10] '] = top1_frame[10],
      [' [11] '] = top1_frame[11],
      [' [12] '] = top1_frame[12],
      [' [13] '] = top1_frame[13],
      [' [14] '] = top1_frame[14],
      [' [15] '] = top1_frame[15],
      [' [16] '] = top1_frame[16],
      [' avg loss '] = loss
   }
   print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy [Center](%%):\t top-1 %.2f\t ',
                       epoch, timer:time().real, loss, top1_frame[16]))
   print('\n')
end -- of test()
-----------------------------------------------------------------------------
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

function testBatch1(inputsCPU, labelsCPU)
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
            top1_frame[j] = top1_frame[j] + 1
         end
      end
   end
   if batchNumber % 128 == 0 then
      print(('Epoch: Testing: [%d][%d/%d], frame[1]: %.2f, frame[8]: %.2f, frame[16]: %.2f'):format(epoch, batchNumber, (nTest/testStride), (top1_frame[1]*100/batchNumber), (top1_frame[8]*100/batchNumber), (top1_frame[16]*100/batchNumber)))
   end
end


-- to test based on one video file
function testBatch2(inputsCPU, labelsCPU)
   batchNumber = batchNumber + 1
   local total = (inputsCPU:size() - 8) / 32 -- tailing frames after n*32+8   
   local candidate = {}
   for i=1,opt.depthSize do
      candidate[i] = 0
   end
   for step=1,total do
      inputs:resize(4,16,3,224,224) --:copy(inputsCPU[{{(step-1)*32+1,step*32+8},{}}])
      labels:resize(4,16,3,224,224) --:copy(labelsCPU[{{(step-1)*32+1,step*32+8},{}}])
      for i=1,4 do
         local begFrame = (step-1)*32 + (i-1)*8 +  1
         local endFrame = (step-1)*32 + i*8     + 16
         inputs[i]:copy(inputsCPU[{{begFrame,endFrame},{}}])
      end
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
         local g = labelsCPU[i][T]
         candidate[pred_sorted[i][T][1]] = candidate[pred_sorted[i][T][1]] + 1
         if pred_sorted[i][T][1] == g then
            top1_center = top1_center + 1
         end
      end
   end
   local _, cand_sorted = candidate:sort(1, true)
   if (cand_sorted[1] == labelsCPU[1][1]) then
      video_acc = video_acc + 1
   end
   if batchNumber % 128 == 0 then
      if opt.debug == true then
         print('label: ', labelsCPU[1][1])
         print('score: ', pred[1][1][labelsCPU[1][1]], 'mean: ', pred[1][1]:mean())
         print('pred: ', pred_sorted[1][1][1], pred_sorted[1][1][2], pred_sorted[1][1][3], pred_sorted[1][1][4], pred_sorted[1][1][5])
      end
      print(('Epoch: Testing: [%d][%d/%d], frame: %.2f, video: %.2f'):format(epoch, batchNumber, (nTest/testStride), (top1_frame*100/T/batchNumber), (top1_center*100/batchNumber)))
   end
end
