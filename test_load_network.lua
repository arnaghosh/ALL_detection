require 'image'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'loadcaffe'

cutorch.setDevice(1)
model = loadcaffe.load('deploy.prototxt', 'bvlc_alexnet.caffemodel', 'cudnn')
--model = nn.Sequential();
--model:add(nn.SpatialConvolution(3,96,11,11,4,4));

for i= 1,10 do
	model:remove()
end
model:add(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1));
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(3,3,2,2));
model:add(nn.View(-1):setNumInputDims(3))
model:add(nn.Linear(12544,4096))
model:add(cudnn.ReLU())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(4096,4096))
model:add(cudnn.ReLU())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(4096,2))
model:add(cudnn.SoftMax())
model:cuda()
print(model)

temp = torch.CudaTensor(10,3,257,257);
out = model:forward(temp);
print(out:size())


--[[
testIm = image.load("patchDataset/Im001_1.jpg");
print(testIm:size())
out = model:forward(testIm:cuda());
print(out:size())
--]]

fileNameLocation = 'Filenames/'
local pfileName = fileNameLocation..'f1_a.txt';
local nfileName = fileNameLocation..'f0_a.txt';
local pfile = io.open(pfileName)
local nfile = io.open(nfileName)
numOfPosImages = 0;
for _ in io.lines(pfileName) do
	numOfPosImages = numOfPosImages+1;
end
numOfNegImages = 0;
for _ in io.lines(nfileName) do
	numOfNegImages = numOfNegImages+1;
end
totImages = numOfPosImages + numOfNegImages;
local AllImages = torch.ByteTensor(totImages,3,257,257);
local AllLabels = torch.Tensor(totImages);
count = 1;
for l in io.lines(pfileName) do
	local img = image.load(l,3,'byte')
	--print(l,img:size())
	AllImages[count]:copy(img);
	AllLabels[count] = 1;
	count = count+1;
	collectgarbage()
end
print("Positive loaded")
for l in io.lines(nfileName) do
	local img = image.load(l,3,'byte')
	AllImages[count]:copy(img);
	AllLabels[count] = 2;
	count = count+1;
	collectgarbage()
end
print("Negative loaded")
local labelsShuffle = torch.randperm(totImages);
local trSize = torch.floor(0.8*totImages);
local teSize = totImages - trSize;

trainData = {
	data = torch.Tensor(trSize,3,257,257),
	labels = torch.Tensor(trSize),
	size = function() return trSize end
}

testData = {
	data = torch.Tensor(teSize,3,257,257),
	labels = torch.Tensor(teSize),
	size = function() return teSize end
}

for i=1,trSize do
	trainData.data[i] = AllImages[labelsShuffle[i]]:type('torch.FloatTensor')
	trainData.labels[i] = AllLabels[labelsShuffle[i]]
end

for i=trSize+1,trSize+teSize do
	testData.data[i-trSize] = AllImages[labelsShuffle[i]]:type('torch.FloatTensor')
	testData.labels[i-trSize] = AllLabels[labelsShuffle[i]]
end


print(sys.COLORS.red ..  '==> preprocessing data: normalize all three channels locally')
local mean = {}
local std = {}

for i=1,3 do
   -- normalize each channel globally:
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end

for i=1,3 do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end

local teMean = {}
local teStd = {}
local trMean = {}
local trStd = {}

for i=1,3 do
   trMean[i] = trainData.data[{ {},i,{},{} }]:mean()
   trStd[i] = trainData.data[{ {},i,{},{} }]:std()
   teMean[i] = testData.data[{ {},i,{},{} }]:mean()
   teStd[i] = testData.data[{ {},i,{},{} }]:std()
end

local criterion = nn.ClassNLLCriterion():cuda()
local trainer = nn.StochasticGradient(model,criterion)
trainer.learningRate = 0.02
trainer.learningRateDecay = 0.001
trainer.shuffleIndices = 0
trainer.maxIteration = 40
batchSize = 64;

collectgarbage()
local iteration =1;
local currentLearningRate = trainer.learningRate;
local input=torch.CudaTensor(batchSize,3,257,257);
local target=torch.CudaTensor(batchSize);
local errorTensor = {}
print(trSize, trSize/batchSize);
print("Training starting")
while true do
	local currentError_ = 0
    for t = 1,math.floor(trSize/batchSize) do
    	local currentError = 0;
      	for t1 = 1,batchSize do
      		t2 = (t-1)*batchSize+t1;
        	target[t1] = trainData.labels[t2];
        	input[t1] = trainData.data[t2]:cuda();
			--print(t1)
        end
        currentError = currentError + criterion:forward(model:forward(input), target)
        --print(currentError)
		currentError_ = currentError_ + currentError*batchSize;
 		model:updateGradInput(input, criterion:updateGradInput(model:forward(input), target))
 		model:accUpdateGradParameters(input, criterion.gradInput, currentLearningRate)
 		print("batch "..t.." done ==>");
 		collectgarbage()
    end
    ---- training on the remaining images, i.e. left after using fixed batch size.
    if(trSize%batchSize ~=0) then
	    local residualInput = torch.CudaTensor(trSize%batchSize,3,257,257);
	    local residualTarget = torch.CudaTensor(trSize%batchSize);

	    for t1=1,(trSize%batchSize) do
	    	t2=batchSize*math.floor(trSize/batchSize) + t1;
	    	residualTarget[t1] = trainData.labels[t2];
	    	residualInput[t1] = trainData.data[t2]:cuda();
		end
		currentError_ = currentError_ + criterion:forward(model:forward(residualInput), residualTarget)*(trSize%batchSize)
		--print("_ "..currentError_);
 		model:updateGradInput(residualInput, criterion:updateGradInput(model:forward(residualInput), residualTarget))
 		model:accUpdateGradParameters(residualInput, criterion.gradInput, currentLearningRate)
 		collectgarbage()
	end
	currentError_ = currentError_ / trSize
	print("#iteration "..iteration..": current error = "..currentError_);
	errorTensor[iteration] = currentError_;
	iteration = iteration + 1
  	currentLearningRate = trainer.learningRate/(1+iteration*trainer.learningRateDecay)
  	if trainer.maxIteration > 0 and iteration > trainer.maxIteration then
    	print("# StochasticGradient: you have reached the maximum number of iterations")
     	print("# training error = " .. currentError_)
     	break
  	end
  	collectgarbage()
end

torch.save("model_7_5.t7",model);
print(errorTensor);

correct = 0
class_perform = {0,0}
class_size = {0,0}
classes = {'Positive', 'Negative'}
for i=1,teSize do
    local groundtruth = testData.labels[i]
    local example = torch.CudaTensor(3,257,257);
    example = testData.data[i]:cuda()
    --print('ground '..groundtruth)
    class_size[groundtruth] = class_size[groundtruth] +1
    local prediction = model:forward(example:cuda())
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
        class_perform[groundtruth] = class_perform[groundtruth] + 1
    end
    collectgarbage()
end
print("Overall correct " .. correct .. " percentage correct" .. (100*correct/teSize) .. " % ")
for i=1,#classes do
   print(classes[i], 100*class_perform[i]/class_size[i] .. " % ")
end



