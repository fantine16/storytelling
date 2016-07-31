require 'nn'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local LSTM = require 'misc.LSTM'


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
	-- create the core lstm network. note +1 for both the START and END tokens
	self.core = LSTM.lstm(self.input_encoding_size, self.vocab_size + 1, self.rnn_size, self.num_layers, dropout)
	self.lookup_table = nn.LookupTable(self.vocab_size + 1, self.input_encoding_size)
	self:_createInitState(1) -- will be lazily resized later during forward passes,改行删掉后，会报错的。
end

function layer:_createInitState(batch_size)
	assert(batch_size ~= nil, 'batch size must be provided')
	-- construct the initial state for the LSTM
	if not self.init_state then self.init_state = {} end -- lazy init
	for h=1,self.num_layers*2 do
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
	local num_lstm=self.images_use_per_story*(self.seq_length+2)
	for t=2,num_lstm do
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
	local imgs=data.images -- imgs的第t个元素是story的第t个图像，每个元素是images tensor，(batch_size*3*224*224)
	local batch_size = data.images[1]:size(1)
	self:_createInitState(batch_size)
	local state = self.init_state

	local sample_max = 1

	local seq = {}
	local seqLogprobs= {}
	for k=1, self.images_use_per_story do
		local seq_t = torch.LongTensor(self.seq_length, batch_size):zero()
		local seqLogprobs_t = torch.FloatTensor(self.seq_length, batch_size)
		for t=1,self.seq_length+2 do
			local xt
			local it
			local ix_t=(k-1)*(self.seq_length+2)+t

			if t==1 then
				xt=imgs[k]
			elseif t==2 then
				it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
				xt = self.lookup_tables[t]:forward(it)
			else
				if sample_max ==1 then
					sampleLogprobs, it = torch.max(logprobs, 2) -- it:
					it = it:view(-1):long() -- ?
				else
					--todo
					;
				end
				xt = self.lookup_table:forward(it)
			end

			if t>=3 then
				--print('LanguageModel 132: it size')
				--print(it:size())
				--print('k = ' .. k .. 't = ' .. t)
				seq_t[t-2]=it -- record the samples
				seqLogprobs_t[t-2] = sampleLogprobs:view(-1):float() -- and also their log likelihoods
			end

			local inputs = {xt,unpack(state)}
			local out = self.core:forward(inputs)
			logprobs = out[self.num_state+1] -- last element is the output vector
			state = {}
			for i=1,self.num_state do table.insert(state, out[i]) end
		end
		table.insert(seq,seq_t)
		table.insert(seqLogprobs,seqLogprobs_t)
	end
	return seq, seqLogprobs
end




function layer:updateOutput(input)
	-- input每个元素是一个完整的story，元素的个数是batch大小
	local imgs=input.images -- imgs的第t个元素是story的第t个图像，每个元素是images tensor，(batch_size*3*224*224)
	local labels=input.labels -- labels的第t个元素是story的第t个标注， 每个元素是 tensor，(batch_size*seq_length)
	local batch_size=imgs[1]:size()[1]
	local seq_length=labels[1]:size()[2]


	if self.clones == nil then self:createClones() end	-- lazily create clones on first forward pass
	self.output:resize((self.seq_length+2)*self.images_use_per_story, batch_size, self.vocab_size+1)
	self:_createInitState(batch_size)

	self.state = {[0] = self.init_state}
	self.inputs = {}
	self.lookup_tables_inputs = {}

	for k=1,self.images_use_per_story do
		for t=1,self.seq_length+2 do
			local xt
			local ix_t=(k-1)*(self.seq_length+2)+t
			if t==1 then
				xt=imgs[k]
			elseif t==2 then
				local it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
				self.lookup_tables_inputs[ix_t] = it
				xt = self.lookup_tables[t]:forward(it) -- NxK sized input (token embedding vectors)
			else
				local it=labels[k][{{},{t-2}}]:clone():resize(labels[k]:size(1))
				it[torch.eq(it,0)] = self.vocab_size + 1 --设置为终止词,end token
				--print(it)
				self.lookup_tables_inputs[ix_t] = it
				xt = self.lookup_tables[ix_t]:forward(it)

			end

			self.inputs[ix_t]={xt, unpack(self.state[ix_t-1])}
			local out = self.clones[ix_t]:forward(self.inputs[ix_t])
			self.output[ix_t]=out[self.num_state+1]
			self.state[ix_t]={} -- the rest is state
			for i=1,self.num_state do table.insert(self.state[ix_t], out[i]) end
			
		end
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
		if t%(self.seq_length+2)==1 then -- 图片
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
	--TODO
	self.gradInput:resizeAs(input):zero() -- reset to zeros
	local L,N,Mp1 = input:size(1), input:size(2), input:size(3)
	local images_per_story = #labels
	local seq_length=labels[1]:size(2) --16
	--print('input size')
	--print(input:size())
	--print('labels size')
	--print(labels[1]:size())
	--print('L=' .. L .. ';N=' .. N .. ';Mp1=' .. Mp1)
	
	local n=0
	local loss=0
	for b=1,N do
		for k=1,images_per_story do
			local first_time=true
			for t=2,seq_length+2 do
				-- fetch the index of the next token in the sequence
				local target_index
				--print('k=' .. k ..';t=' .. t)
				if t-1>seq_length then
					target_index = 0
					--print('x')
				else
					target_index = labels[k][{b,t-1}]
					--print('y')
					--print(target_index)
				end
				-- the first time we see null token as next index, actually want the model to predict the END token
				if target_index == 0 and first_time then
					target_index=Mp1
					first_time=false
				end

				-- if there is a non-null next token, enforce loss!
				if target_index ~= 0 then
					-- accumulate loss
					--print((k-1)*(seq_length+2)+t)
					--print(b)
					--print(target_index)
					loss=loss - input[{(k-1)*(seq_length+2)+t,b,target_index}]
					self.gradInput[{(k-1)*(seq_length+2)+t,b,target_index}] = -1
					n = n + 1
				end
			end
		end
	end

	self.output = loss/n -- normalize by number of predictions that were made
	self.gradInput:div(n)
	return self.output
end

function crit:updateGradInput(input, seq)
	return self.gradInput
end
