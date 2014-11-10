require 'nn'
require 'optim'
require 'xlua'

dofile 'loadTensor.lua'


opt = {}
weightDecay = 1e-2
opt.learningRate = 1e-3
opt.weightDecay = 0
opt.momentum = 0
opt.save = 'results'
opt.batchSize = 1


train_file_questions = "../data/questions.train.txt"
train_file_answers = "../data/answers.train.txt"

test_file_questions = "../data/questions.test.txt"
test_file_answers = "../data/answers.test.txt"



train_questions = readData(train_file_questions)
m = train_questions:size(1)
ninputs = train_questions:size(2)
train_answers = torch.reshape(readData(train_file_answers):int(), m)

test_questions = readData(test_file_questions)
mtest = test_questions:size(1)
test_answers = torch.reshape(readData(test_file_answers):int(), mtest)



trainData = {data = train_questions, labels = train_answers} 
testData = {data = test_questions, labels = test_answers} 



--We normalize the data
mean_data = trainData.data:mean(1)
std_data = trainData.data:std(1)

--We save the mean and the std of each composant
local filename = paths.concat(opt.save, 'params')
os.execute('mkdir -p ' .. sys.dirname(filename))
print('==> saving mean and std to '..filename)
torch.save(filename, {mean=mean_data, std=std_data})



for i = 1, ninputs do
	mean = trainData.data[{{}, i}]:mean()
	std = trainData.data[{{}, i}]:std()

	trainData.data[{{}, i}]:add(-mean)
	testData.data[{{}, i}]:add(-mean)
	if std > 0 then
		trainData.data[{{}, i}]:div(std)
		testData.data[{{}, i}]:div(std)
	end
end




noutputs = 4

model = nn.Sequential()
model:add(nn.Reshape(ninputs))
model:add(nn.Linear(ninputs, noutputs))
model:add( nn.LogSoftMax() )


--model:add( nn.Linear(ninputs,10) )
--model:add( nn.Tanh() )
--model:add( nn.Linear(10,noutputs) )
--model:add( nn.LogSoftMax() )



criterion = nn.ClassNLLCriterion()


classes = {'1','2','3','4'}

confusion = optim.ConfusionMatrix(classes)

reg = {}
reg[1] = model:get(2).weight
reg[2] = model:get(2).bias




if model then
   parameters,gradParameters = model:getParameters()
end

--Methode d'optimisation: SGD

 optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
   }
   optimMethod = optim.sgd

trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))


function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(m)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

   for t = 1,m,opt.batchSize do
      -- disp progress
      xlua.progress(t, m)

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,m) do
         -- load new sample
         local input = trainData.data[shuffle[i]]
         local target = trainData.labels[shuffle[i]]


         table.insert(inputs, input)
         table.insert(targets, target)
      end
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
	       -- get new parameters
	       if x ~= parameters then
	          parameters:copy(x)
	       end

	       -- reset gradients
	       gradParameters:zero()

	       -- f is the average of all criterions
	       local f = 0

	       -- evaluate function for complete mini batch
	       for i = 1,#inputs do
	          -- estimate f
	          local output = model:forward(inputs[i])
	          local err = criterion:forward(output, targets[i])
	          f = f + err

	          -- estimate df/dW
	          local df_do = criterion:backward(output, targets[i])
	          model:backward(inputs[i], df_do)

	          -- update confusion
	          confusion:add(output, targets[i])
	       end

	       -- normalize gradients and f(X)
	       gradParameters:div(#inputs)
	       f = f/#inputs

	       -- return f and df/dX
	       return f,gradParameters
	    end

      -- optimize on current mini-batch
      if optimMethod == optim.asgd then
         _,_,average = optimMethod(feval, parameters, optimState)
      else
         optimMethod(feval, parameters, optimState)
      end
   end

   --REGULARIZATION
   for _,w in ipairs(reg) do
      w:add(-weightDecay, w)
   end

   -- time taken
   time = sys.clock() - time
   time = time / m
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   -- save/log current net
   local filename = paths.concat(opt.save, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end


function test()
   -- local vars
   local time = sys.clock()

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')
   for t = 1,mtest do
      -- disp progress
      xlua.progress(t, mtest)

      -- get new sample
      local input = testData.data[t]

      local target = testData.labels[t]

      -- test sample
      local pred = model:forward(input)
      confusion:add(pred, target)
   end

   -- timing
   time = sys.clock() - time
   time = time / mtest
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end
   
   -- next iteration:
   confusion:zero()
end