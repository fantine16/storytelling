require 'torch'
require 'nn'
require 'nngraph'
-- exotic things
require 'loadcaffe'
-- local imports
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
require 'misc.LanguageModel'
require 'misc.optim_updates'
require 'misc.DataLoader'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an StoryTelling model')
cmd:text()
cmd:text('Options')
-- Data input settings
cmd:option('-input_h5','dataset/storytelling.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','dataset/storytelling.json','path to the json file containing additional info and vocab')
cmd:option('-cnn_proto','model/VGG_ILSVRC_16_layers_deploy.prototxt','path to CNN prototxt file in Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-cnn_model','model/VGG_ILSVRC_16_layers.caffemodel','path to CNN model file containing the weights, Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
-- Model settings
cmd:option('-rnn_size',768,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-input_encoding_size',768,'the encoding size of each token in the vocabulary, and the image.')
-- Optimization: General
cmd:option('-max_epoch',15, 'max number of epoch to run for (-1 = run forever)')
cmd:option('-max_iters',-1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size',32,'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-images_per_story',5,'number of images for each story during training.')
cmd:option('-images_use_per_story', 5)
cmd:option('-finetune_cnn_after', -1, 'After what iteration do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-drop_prob_lm', 0.5, 'strength of dropout in the Language Model RNN')
-- Optimization: for the Language Model
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-num_layers',1,'number of hidden layers of rnn')
cmd:option('-learning_rate',4e-4,'learning rate')
cmd:option('-learning_rate_decay_start', 20000, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 30000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')
-- Optimization: for the CNN
cmd:option('-cnn_optim','adam','optimization to use for CNN')
cmd:option('-cnn_optim_alpha',0.8,'alpha for momentum of CNN')
cmd:option('-cnn_optim_beta',0.999,'alpha for momentum of CNN')
cmd:option('-cnn_learning_rate',1e-5,'learning rate for the CNN')
cmd:option('-cnn_weight_decay', 0, 'L2 weight decay just for the CNN')


-- Evaluation/Checkpointing
cmd:option('-val_images_use', 64, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-losses_log_every', 1000, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
cmd:option('-save_checkpoint_every', 1000,'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', '', 'folder to save checkpoints into (empty = this folder)')

cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')

cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-debug', true)

cmd:text()

opt = cmd:parse(arg)

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end
print(opt)
-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}

-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------

local protos = {}
  -- create protos from scratch
  -- intialize language model
if string.len(opt.start_from) > 0 then
	print('initializing weights from ' .. opt.start_from)
	local loaded_checkpoint = torch.load(opt.start_from)
	protos = loaded_checkpoint.protos
	net_utils.unsanitize_gradients(protos.cnn)
	local lm_modules = protos.lm:getModulesList()
	for k,v in pairs(lm_modules) do net_utils.unsanitize_gradients(v) end
	protos.crit = nn.LanguageModelCriterion() -- not in checkpoints, create manually
else
	local lmOpt = {}
	lmOpt.vocab_size = loader:getVocabSize()
	lmOpt.input_encoding_size = opt.input_encoding_size
	lmOpt.rnn_size = opt.rnn_size
	lmOpt.num_layers = opt.num_layers
	lmOpt.dropout = opt.drop_prob_lm
	lmOpt.seq_length = loader:getSeqLength()
	lmOpt.batch_size = opt.batch_size * opt.images_per_story
	lmOpt.images_use_per_story = opt.images_use_per_story
	print(lmOpt)
	protos.lm = nn.LanguageModel(lmOpt)
	-- initialize the ConvNet
	local cnn_backend = opt.backend
	local cnn_raw = loadcaffe.load(opt.cnn_proto, opt.cnn_model, cnn_backend)
	protos.cnn = net_utils.build_cnn(cnn_raw, {encoding_size = opt.input_encoding_size, backend = cnn_backend})
	-- criterion for the language model
	protos.crit = nn.LanguageModelCriterion()
end


if opt.gpuid >= 0 then
	for k,v in pairs(protos) do v:cuda() end
end

-- flatten and prepare all model parameters to a single vector. 
-- Keep CNN params separate in case we want to try to get fancy with different optims on LM/CNN
local params, grad_params = protos.lm:getParameters()
local cnn_params, cnn_grad_params = protos.cnn:getParameters()
print('total number of parameters in LM: ', params:nElement())
print('total number of parameters in CNN: ', cnn_params:nElement())
assert(params:nElement() == grad_params:nElement())
assert(cnn_params:nElement() == cnn_grad_params:nElement())
--assert(false)

local thin_lm = protos.lm:clone()
thin_lm.core:share(protos.lm.core, 'weight', 'bias') -- TODO: we are assuming that LM has specific members! figure out clean way to get rid of, not modular.
thin_lm.lookup_table:share(protos.lm.lookup_table, 'weight', 'bias')
local thin_cnn = protos.cnn:clone('weight', 'bias')
-- sanitize all modules of gradient storage so that we dont save big checkpoints
net_utils.sanitize_gradients(thin_cnn)
local lm_modules = thin_lm:getModulesList()
for k,v in pairs(lm_modules) do net_utils.sanitize_gradients(v) end
-- create clones and ensure parameter sharing. we have to do this 
-- all the way here at the end because calls such as :cuda() and
-- :getParameters() reshuffle memory around.
protos.lm:createClones()

collectgarbage() -- "yeah, sure why not"


-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
	local verbose = utils.getopt(evalopt, 'verbose', true)
	local val_images_use = utils.getopt(evalopt, 'val_images_use', 2000)

	local n = 0 --处理的图像的数量
	local loss_sum = 0
	local loss_evals = 0
	local predictions={} --TODO
	local lang_stats --TODO
	protos.cnn:evaluate()
	protos.lm:evaluate()
	loader:resetIterator(split)
	local vocab = loader:getVocab()	

	while true do 
		-- fetch a batch of data
		local data = loader:getBatch{batch_size = opt.batch_size, split = split, images_per_story = opt.images_per_story, images_use_per_story = opt.images_use_per_story}
		n=n+data.images[1]:size(1)
		for k,v in pairs(data.images) do
			data.images[k]=net_utils.prepro(data.images[k], true, opt.gpuid>=0) --for循环，可能比较慢
		end	
		-- forward the ConvNet on images (most work happens here)
		for k,v in pairs(data.images) do
			data.images[k] = protos.cnn:forward(data.images[k])
		end
		local logprobs = protos.lm:forward(data)
		local loss = protos.crit:forward(logprobs, data.labels)
		loss_sum = loss_sum + loss
		loss_evals = loss_evals + 1

		-- forward the model to also get generated samples for each image
		local seq = protos.lm:sample(data)
		print(torch.type(seq))
		print(seq:size())
		local sents = net_utils.decode_sequence(vocab, seq)

		for k=1, opt.batch_size do
			local entry = {story_id = data.infos[k], caption = sents[k]}
			table.insert(predictions, entry)
			print(string.format('story %s: %s', entry.story_id, entry.caption))
		end

		if loss_evals % 10 == 0 then collectgarbage() end
		if n >= val_images_use then break end -- we've used enough images
	end
	return loss_sum/loss_evals, predictions, lang_stats
end


-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local iter = 1
local function lossFun()
	protos.cnn:training()
	protos.lm:training()
	grad_params:zero() --每次反向传播梯度都是重新计算的，所以这里置零并不影响。但是为什么这么做？
	if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
		cnn_grad_params:zero()
	end
	-----------------------------------------------------------------------------
	-- Forward pass
	-----------------------------------------------------------------------------
	-- get batch of data
	local data = loader:getBatch{batch_size = opt.batch_size, split = 'train', images_per_story = opt.images_per_story, images_use_per_story = opt.images_use_per_story}
	
	data.raw_images={}
	for k,v in pairs(data.images) do
		data.images[k]=net_utils.prepro(data.images[k], true, opt.gpuid>=0) --for循环，可能比较慢
	end	
	-- forward the ConvNet on images (most work happens here)
	for k,v in pairs(data.images) do
		data.raw_images[k]=data.images[k]:clone() -- raw_images存原始的(N,3,224,224)的图像tensor
		data.images[k] = protos.cnn:forward(data.images[k])
	end
	-- forward the language model
	local logprobs = protos.lm:forward(data)
	-- forward the language model criterion
	local loss = protos.crit:forward(logprobs, data.labels)
	-----------------------------------------------------------------------------
	-- Backward pass
	-----------------------------------------------------------------------------
	-- backprop criterion
	local dlogprobs = protos.crit:backward(logprobs, data.labels)
	-- backprop language model
	local dimgs = protos.lm:backward(data, dlogprobs)
	--print(torch.type(dimgs))
	--print('dimgs num = ' .. #dimgs)
	-- backprop the CNN, but only if we are finetuning
	if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
		for i=#dimgs,1 ,-1 do
			protos.cnn:backward(data.raw_images[i], dimgs[i])
		end
	end
	--TODO
	-- clip gradients
	-- opt.grad_clip为0.1。把梯度截取到-0.1和0.1之间。这样做是为了优化吗？
	grad_params:clamp(-opt.grad_clip, opt.grad_clip)
	-- apply L2 regularization
	if opt.cnn_weight_decay > 0 then
		cnn_grad_params:add(opt.cnn_weight_decay, cnn_params)
		-- note: we don't bother adding the l2 loss to the total loss, meh.
		cnn_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
	end
	
	return loss	

end

--assert(false)

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss0
local optim_state = {}
local cnn_optim_state = {}
local loss_history = {}
local val_lang_stats_history = {}
local val_loss_history = {}
local best_score
local num_train = loader:getTrainNum()
while true do	
	-- eval loss/gradient
	local loss = lossFun()
	
	if iter % opt.losses_log_every == 0 then loss_history[iter] = loss end
	--print(torch.type(iter))
	--print(torch.type(loss))
	print(string.format('epoch %.2f , iter %d: %f', iter*opt.batch_size/num_train, iter, loss))

	if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then
		-- evaluate the validation performance
		local val_loss, val_predictions, lang_stats = eval_split('val', {val_images_use = opt.val_images_use})
		print('validation loss: ', val_loss)
		--TODO
		--print(lang_stats)
		val_loss_history[iter] = val_loss

		local checkpoint_path = path.join(opt.checkpoint_path, 'model_id' .. opt.id)

		-- write a (thin) json report
		local checkpoint = {}
		checkpoint.opt = opt
		checkpoint.iter = iter
		checkpoint.loss_history = loss_history
		checkpoint.val_loss_history = val_loss_history
		checkpoint.val_predictions = val_predictions -- save these too for CIDEr/METEOR/etc eval

		utils.write_json(checkpoint_path .. '.json', checkpoint)
		print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

		-- write the full model checkpoint as well if we did better than ever
		local current_score
		if lang_stats then
			-- use CIDEr score for deciding how well we did
			current_score = lang_stats['CIDEr']
		else
			-- use the (negative) validation loss as a score
			current_score = -val_loss
		end

		if best_score == nil or current_score > best_score then
			best_score = current_score
			if iter > 0 then
				local save_protos = {}
				save_protos.lm = thin_lm -- these are shared clones, and point to correct param storage
				save_protos.cnn = thin_cnn
				checkpoint.protos = save_protos
				-- also include the vocabulary mapping so that we can use the checkpoint 
				-- alone to run on arbitrary images without the data loader
				checkpoint.vocab = loader:getVocab()
				torch.save(checkpoint_path .. '.t7', checkpoint)
				print('wrote checkpoint to ' .. checkpoint_path .. '.t7')
			end
		end
	end


	-- decay the learning rate for both LM and CNN
	local learning_rate = opt.learning_rate
	local cnn_learning_rate = opt.cnn_learning_rate
	if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
		local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
		local decay_factor = math.pow(0.5, frac)
		learning_rate = learning_rate * decay_factor -- set the decayed rate
		cnn_learning_rate = cnn_learning_rate * decay_factor
	end


	-- perform a parameter update
	if opt.optim == 'rmsprop' then
		rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
	elseif opt.optim == 'adagrad' then
		adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
	elseif opt.optim == 'sgd' then
		sgd(params, grad_params, opt.learning_rate)
	elseif opt.optim == 'sgdm' then
		sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
	elseif opt.optim == 'sgdmom' then
		sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
	elseif opt.optim == 'adam' then
		adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
	else
		error('bad option opt.optim')
	end

	-- do a cnn update (if finetuning, and if rnn above us is not warming up right now)
	if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
		if opt.cnn_optim == 'sgd' then
			sgd(cnn_params, cnn_grad_params, cnn_learning_rate)
		elseif opt.cnn_optim == 'sgdm' then
			sgdm(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, cnn_optim_state)
		elseif opt.cnn_optim == 'adam' then
			adam(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, cnn_optim_state)
		else
			error('bad option for opt.cnn_optim')
		end
	end

	-- stopping criterions
	iter = iter + 1
	if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
	if loss0 == nil then loss0 = loss end
	if loss > loss0 * 20 then
		print('loss seems to be exploding, quitting.')
		break
	end
	if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion
	if opt.max_epoch > 0 and iter*opt.batch_size/num_train >= opt.max_epoch then break end -- stopping criterion
	
end