require 'image'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'loadcaffe'

model = torch.load('model_7_5.t7');
for i=1,10 do
	model:remove()
end


--input = image.load('Augmented_Dataset/1_0.jpg')
fileNameLocation = 'Filenames/'
local fileName = fileNameLocation..'whole_slide.txt';
local file = io.open(fileName)

for l in io.lines(fileName) do
	input = image.load(l,3)
	out = model:forward(input:cuda());
	out_GAP = torch.cumsum(out,1)
	out_GAP = out_GAP[256];

	min = out_GAP:min();
	max = out_GAP:max();
	--print(min,max)
	out_GAP:add(-min);
	out_GAP:div(max-min);
	--print("hihi");
	l1 = string.gsub(l,"ALL_IDB dataset/ALL_IDB1/img/", "Deploy_Results/filter")
	print(l1)
	image.save(l1,out_GAP)
end