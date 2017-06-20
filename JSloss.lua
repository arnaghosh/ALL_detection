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
local criterion = nn.DistKLDivCriterion():cuda()

--input = image.load('Augmented_Dataset/1_0.jpg')
z = torch.Tensor(3,257,257):fill(1);
out = model:forward(z:cuda());
a= torch.Tensor(256,256);
y =torch.Tensor();

for i=1,256 do
   for j=i,256 do

      y1 = torch.Tensor(out[i]:size()):copy(out[i])
      y2 = torch.Tensor(out[j]:size()):copy(out[j])
      y1:mul(0.5)
      y2:mul(0.5)
      torch.add(y, y1, y2)
      M= torch.log(y)
      loss1 = criterion:forward(M:cuda(), out[j]:cuda())
      loss2 = criterion:forward(M:cuda(),out[i]:cuda())
      JSloss = 0.5*loss1 + 0.5*loss2
      a[i][j] = JSloss
      a[j][i] = JSloss
   end

   
end

min = a:min();
max = a:max();
--print(min,max)
a:add(-min);
a:div(max-min);
A ,I = torch.sort(a[1])
X =torch.Tensor(256,256);
for i=1,256 do
   X[{{},{i}}]= a[{{},{I[i]}}];
end
image.save('similarity.jpg', a);
image.save('similaritySorted.jpg',X)
print(X[1])
--print(a);
