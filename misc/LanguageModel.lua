require 'nn'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local LSTM = require 'misc.LSTM'
local GRU = require 'misc.GRU'
local RNN = require 'misc.RNN'


-------------------------------------------------------------------------------
-- Language Model core
-------------------------------------------------------------------------------
local layer, parent = torch.class('nn.LanguageModel', 'nn.Module')
function layer:__init(opt)
	parent.__init(self)

	-- options for core network
	self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
	self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
	self.rnn_size = utils.getopt(opt, 'rnn_size')
	self.num_layers = utils.getopt(opt, 'num_layers', 1)
	self.images_per_story = utils.getopt(opt,'images_per_story',5)
	self.images_use_per_story = utils.getopt(opt,'images_use_per_story',5)
	local dropout = utils.getopt(opt, 'dropout', 0)
	-- options for Language Model
	self.seq_length = utils.getopt(opt, 'seq_length')
	self.rnn_type=opt.rnn_type
	-- create the core lstm network. note +1 for both the START and END tokens
	--单词表添加"停止词"和"逗号"

	if self.rnn_type == 'lstm' then
		self.core = LSTM.lstm(self.input_encoding_size, self.vocab_size + 2, self.rnn_size, self.num_layers, dropout)
	elseif self.rnn_type == 'gru' then
		self.core = GRU.gru(self.input_encoding_size, self.vocab_size + 2, self.rnn_size, self.num_layers, dropout)
	elseif self.rnn_type == 'rnn' then 
		self.core = RNN.rnn(self.input_encoding_size, self.vocab_size + 2, self.rnn_size, self.num_layers, dropout)
	end

	self.lookup_table = nn.LookupTable(self.vocab_size + 2, self.input_encoding_size)
	self:_createInitState(1) -- will be lazily resized later during forward passes,改行删掉后，会报错的。
end

function layer:_createInitState(batch_size)
	assert(batch_size ~= nil, 'batch size must be provided')
	-- construct the initial state 
	if not self.init_state then self.init_state = {} end
	--self.init_state = {} 


	local num_layers 
	if self.rnn_type=='lstm' then
		num_layers=self.num_layers*2
	else
		num_layers=self.num_layers
	end

	for h=1,num_layers do
		-- note, the init state Must be zeros because we are using init_state to init grads in backward call too
		if self.init_state[h] then
			if self.init_state[h]:size(1) ~= batch_size then
				self.init_state[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
			end
		else
			self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
		end
	end
	self.num_state = #self.init_state



	--print(torch.type(self.init_state[1]))
	--print(torch.type(self.init_state[2]))
	--print('!!!')
end

function layer:parameters()
	-- we only have two internal modules, return their params
	local p1,g1 = self.core:parameters()
	local p2,g2 = self.lookup_table:parameters()

	local params = {}
	for k,v in pairs(p1) do table.insert(params, v) end
	for k,v in pairs(p2) do table.insert(params, v) end
	
	local grad_params = {}
	for k,v in pairs(g1) do table.insert(grad_params, v) end
	for k,v in pairs(g2) do table.insert(grad_params, v) end

	-- todo: invalidate self.clones if params were requested?
	-- what if someone outside of us decided to call getParameters() or something?
	-- (that would destroy our parameter sharing because clones 2...end would point to old memory)

	return params, grad_params
end

function layer:createClones()
	-- construct the net clones
	print('constructing clones inside the LanguageModel')
	self.clones = {self.core}
	self.lookup_tables = {self.lookup_table}
	local num_lstm=5+1+5*(self.seq_length+1)
	for t=2,num_lstm do
		--print(t)
		self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
		self.lookup_tables[t] = self.lookup_table:clone('weight', 'gradWeight')
	end
end

function layer:getModulesList()
	return {self.core, self.lookup_table}
end

function layer:training()
	if self.clones == nil then self:createClones() end -- create these lazily if needed
	for k,v in pairs(self.clones) do v:training() end
	for k,v in pairs(self.lookup_tables) do v:training() end
end

function layer:evaluate()
	if self.clones == nil then self:createClones() end -- create these lazily if needed
	for k,v in pairs(self.clones) do v:evaluate() end
	for k,v in pairs(self.lookup_tables) do v:evaluate() end
end

--[[
takes a batch of images and runs the model forward in sampling mode
Careful: make sure model is in :evaluate() mode if you're calling this.
--]]
function layer:sample(data, opt)

	local debug = false
	local function dprint(sth)
		if debug then
			print(sth)
		end
	end
	local function dassert(flag,sth)
		if debug then
			assert(flag, sth)
		end
	end

	local sample_max = utils.getopt(opt, 'sample_max', 1)
	local beam_size = utils.getopt(opt, 'beam_size', 1)
	local temperature = utils.getopt(opt, 'temperature', 1.0)
	local labels = data.labels
	local seq_length=labels[1]:size(2)

	local imgs=data.images -- imgs的第t个元素是story的第t个图像，每个元素是images tensor，(batch_size*3*224*224)
	local batch_size = data.images[1]:size(1)
	self:_createInitState(batch_size)
	local state = self.init_state

	local seq = torch.LongTensor(5*(seq_length+1),batch_size):zero()
	local seqLogprobs= torch.FloatTensor(5*(seq_length+1),batch_size)
	local logprobs --(batch_size, vocab_size+2), (4,9772)
	
	for t=1,5+1+5*(seq_length+1) do
		if t<=5 then
			xt=imgs[t]
		elseif t==6 then
			it=torch.LongTensor(batch_size):fill(self.vocab_size+2) --终止词
			xt = self.lookup_table:forward(it)
		else
			if sample_max==1 then
				sampleLogprobs, it = torch.max(logprobs, 2)
				it = it:view(-1):long()
			else
				local prob_prev
				if temperature == 1.0 then
					prob_prev = torch.exp(logprobs) -- fetch prev distribution: shape Nx(M+1)
				else
					prob_prev = torch.exp(torch.div(logprobs, temperature))
				end
				it = torch.multinomial(prob_prev, 1)
				sampleLogprobs = logprobs:gather(2, it)
				it = it:view(-1):long()
			end
		end


		if t>=7 then
			seq[t-6]=it
			seqLogprobs[t-6] = sampleLogprobs:view(-1):float()
		end

		local inputs = {xt,unpack(state)}
		local out = self.core:forward(inputs)
		logprobs = out[self.num_state+1]
		dprint('logprobs size:')
		dprint(logprobs:size())
		dassert(false)
		state = {}
		for i=1,self.num_state do table.insert(state, out[i]) end
	end

	return seq, seqLogprobs
end




function layer:updateOutput(input)
	local debug = false
	local function dprint(sth)
		if debug then
			print(sth)
		end
	end
	dprint('layer:updateOutput :')
	-- input每个元素是一个完整的story，元素的个数是batch大小
	local imgs=input.images -- imgs的第t个元素是story的第t个图像，每个元素是images tensor，(batch_size*3*224*224)
	local labels=input.labels -- labels的第t个元素是story的第t个标注， 每个元素是 tensor，(batch_size*seq_length)
	local batch_size=imgs[1]:size()[1]
	local seq_length=labels[1]:size()[2]
	local pro_label = torch.LongTensor(batch_size, 5*(seq_length+1)):fill(0)
	dprint(labels)
	for i=1,#labels do
		dprint(labels[i])
	end
	for i=1, batch_size do
		local t=torch.LongTensor{self.vocab_size+2} --停止词,但是并没有加入到pro_label中
		for j=1, self.images_use_per_story do
			local temp=labels[j][i]
			dprint(temp)
			temp=temp[torch.ne(temp,0)]
			dprint(temp)
			t=torch.cat(t,temp)
			t=torch.cat(t,torch.LongTensor{self.vocab_size+1}) --逗号
			--print(t)
		end
		dprint(t)
		pro_label[i][{{1,t:size(1)-1}}]:copy(t[{{2,t:size(1)}}])--pro_label中,每句话的第一个词不是停止词
		pro_label[torch.eq(pro_label,0)]=self.vocab_size+2--停止词
	end
	pro_label=pro_label:t()
	dprint(pro_label)
	--assert(false)
	if self.clones == nil then self:createClones() end	-- lazily create clones on first forward pass
	self.output:resize(5+1+5*(self.seq_length+1), batch_size, self.vocab_size+2)
	self:_createInitState(batch_size)

	self.state = {[0] = self.init_state}
	self.inputs = {}
	self.lookup_tables_inputs = {}
	
	for t=1, 5+1+5*(self.seq_length+1) do
		local xt
		if t<=5 then
			xt = imgs[t]
		elseif t==6 then
			local it = torch.LongTensor(batch_size):fill(self.vocab_size+2) --设置为终止词,end token
			self.lookup_tables_inputs[t] = it
			xt = self.lookup_tables[t]:forward(it)
		else
			local it =  pro_label[t-6]:clone()
			it[torch.eq(it,0)] = self.vocab_size + 2 --设置为终止词,end token
			self.lookup_tables_inputs[t] = it
			xt = self.lookup_tables[t]:forward(it)
		end
		-- construct the inputs
		self.inputs[t] = {xt,unpack(self.state[t-1])}
		--print(t)
		--print(xt:size())
		--print(self.state[t-1][1]:size())
		--print(self.state[t-1][2]:size())
		--assert(false)
		-- forward the network
		--print(#self.inputs[t])
		local out = self.clones[t]:forward(self.inputs[t])
		-- process the outputs
		self.output[t] = out[self.num_state+1] 
		self.state[t] = {}
		for i=1,self.num_state do table.insert(self.state[t], out[i]) end

	end

	return self.output --

end

function layer:updateGradInput(input, gradOutput) --gradOutput dim: (90,10,9771)
	local dimgs={} -- grad on input images
	-- go backwards and lets compute gradients
	local batch_size=gradOutput:size(2) -- 10
	local dstate = {[gradOutput:size(1)] = self.init_state} -- why? 90
	for t = gradOutput:size(1),1,-1 do
		--print('LanguageModel.lua 227; t=' .. t)
		local dout ={}
		for k=1,#dstate[t] do table.insert(dout, dstate[t][k]) end
		table.insert(dout,gradOutput[t])
		local dinputs = self.clones[t]:backward(self.inputs[t], dout)
		local dxt = dinputs[1] -- first element is the input vector
		dstate[t-1] = {}
		for k=2,self.num_state+1 do table.insert(dstate[t-1], dinputs[k]) end
		if t<=5 then -- 图片
			table.insert(dimgs,dxt)
		else
			local it = self.lookup_tables_inputs[t]
			--print(it:size())
			--print(it)
			--print('it type: ' .. torch.type(it))
			--print(dxt:size())
			--print('dxt type: ' .. torch.type(dxt))
			self.lookup_tables[t]:backward(it:cuda(), dxt) -- backprop into lookup table
		end
	end
	self.gradInput = dimgs
	return self.gradInput


end

-------------------------------------------------------------------------------
-- Language Model-aware Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.LanguageModelCriterion', 'nn.Criterion')
function crit:__init()
	parent.__init(self)
end

function crit:updateOutput(input, labels)
	local debug = false
	local function dprint(sth)
		if debug then
			print(sth)
		end
	end
	local batch_size=input:size(2)
	local seq_length=labels[1]:size(2)
	local vocab_size=input:size(3)-2
	local images_use_per_story = #labels
	dprint("input size:")
	dprint(input:size())
	dprint('batch_size = ' .. batch_size)
	dprint('seq_length = ' .. seq_length)
	local pro_label = torch.LongTensor(batch_size, 5*(seq_length+1)+1):fill(0)
	for i=1, batch_size do
		local t=torch.LongTensor{vocab_size+2} --停止词
		for j=1, images_use_per_story do
			local temp=labels[j][i]
			temp=temp[torch.ne(temp,0)]
			t=torch.cat(t,temp)
			t=torch.cat(t,torch.LongTensor{vocab_size+1}) --逗号
			--print(t)
		end
		--print(t)
		pro_label[i][{{1,t:size(1)-1}}]:copy(t[{{2,t:size(1)}}])--pro_label中,每句话的第一个词不是停止词
		pro_label[torch.eq(pro_label,0)]=vocab_size+2
	end
	--assert(false)
	pro_label=pro_label:t()
	dprint("pro_label:size() ")
	dprint(pro_label:size())
	dprint(pro_label)

	--TODO
	self.gradInput:resizeAs(input):zero() -- reset to zeros
	local L,N,Mp1 = input:size(1), input:size(2), input:size(3)
	--local D=pro_label:size(1)
	Mp1=Mp1-1
	local images_per_story = #labels
	local seq_length=labels[1]:size(2)	
	local n=0
	local loss=0
	dprint("loss type: ")
	dprint(torch.type(loss))
	for b=1,N do
		local first_time=true
		for t=6,L do	
			local target_index		
			if t==L then
				target_index=0
			else
				target_index=pro_label[{t-5,b}]
			end
			if target_index==0 and first_time then
				target_index=Mp1
				first_time=false
			end

			if target_index~=0 then
				loss = loss-input[{t,b,target_index}]				
				dprint('t=' .. t)
				dprint('b=' .. b)
				dprint('target_index=' .. target_index)
				dprint(torch.type(input[{t,b,target_index}]))
				self.gradInput[{t,b,target_index}] = -1
				n=n+1
			end			
		end
	end

	self.output = loss/n -- normalize by number of predictions that were made
	self.gradInput:div(n)
	dprint(torch.type(self.output))
	dprint(torch.type(loss/n))
	dprint(torch.type(loss))
	dprint(torch.type(n))
	return self.output
end

function crit:updateGradInput(input, seq)
	return self.gradInput
end
