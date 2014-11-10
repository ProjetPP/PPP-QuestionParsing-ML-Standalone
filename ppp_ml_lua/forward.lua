require 'nn'
require 'optim'


dofile 'loadTensor.lua'


model = torch.load('results/model.net', 'binary')
params = torch.load('results/params', 'binary')
file = '../data/input.txt'
output = '../data/output.txt'

data = readData(file)

nb_words = data:size(1)
ninputs = data:size(2)


for i = 1, ninputs do
	mean = params.mean[1][i]
	std = params.std[1][i]

	for j = 1,nb_words do
		data[j][i] = data[j][i] - mean
		if std > 0 then
			data[j][i] = data[j][i]/std
		end
	end
end





f = io.open(output, 'w')

for i =1,nb_words do
	output = model:forward(data[i])
	y,i=torch.max(output,1)
	f:write(i[1], '\n')

end

f:close()
