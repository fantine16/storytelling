require 'torch'
require 'nn'
require 'nngraph'
-- exotics
require 'loadcaffe'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.LanguageModel'
local net_utils = require 'misc.net_utils'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')

-- Input paths
cmd:option('-model','model_id.t7','path to model to evaluate')
-- Basic options
cmd:option('-batch_size', 15, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-num_stories', 1000, 'how many images to use when periodically evaluating the loss? (-1 = all)')
cmd:option('-input_h5','dataset/storytelling.h5','path to the h5file containing the preprocessed dataset. empty = fetch from model checkpoint.')
cmd:option('-input_json','dataset/storytelling.json','path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
cmd:option('-split', 'test', 'if running on MSCOCO images, which split to use: val|test|train')
-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', 'evalscript', 'an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Load the model checkpoint to evaluate
-------------------------------------------------------------------------------
assert(string.len(opt.model) > 0, 'must provide a model')
local checkpoint = torch.load(opt.model)
-- override and collect parameters
if string.len(opt.input_h5) == 0 then opt.input_h5 = checkpoint.opt.input_h5 end
if string.len(opt.input_json) == 0 then opt.input_json = checkpoint.opt.input_json end
if opt.batch_size == 0 then opt.batch_size = checkpoint.opt.batch_size end
local fetch = {'rnn_size', 'input_encoding_size', 'drop_prob_lm', 'cnn_proto', 'cnn_model', 'images_per_story'}
for k,v in pairs(fetch) do 
  opt[v] = checkpoint.opt[v] -- copy over options from model
end
print(opt)


local vocab = checkpoint.vocab -- ix -> word mapping

local loader
loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}
-------------------------------------------------------------------------------
-- Load the networks from model checkpoint
-------------------------------------------------------------------------------
local protos = checkpoint.protos
protos.lm:createClones() -- reconstruct clones inside the language model
protos.crit = nn.LanguageModelCriterion()
if opt.gpuid >= 0 then for k,v in pairs(protos) do v:cuda() end end
-------------------------------------------------------------------------------
-- Evaluation fun(ction)
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
	local verbose = utils.getopt(evalopt, 'verbose', true)
	local num_stories = utils.getopt(evalopt, 'num_stories', 1000)
	
	local n = 0 --处理的图像的数量
	local loss_sum = 0
	local loss_evals = 0
	protos.cnn:evaluate()
	protos.lm:evaluate()
	loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
	local predict = {}

	while true do 
		local data = loader:getBatch{batch_size = opt.batch_size, split = split, images_per_story = opt.images_per_story}
		--utils.batch_equal(data) -- 判断data是否正确，两两判断是否相同
		data.raw_images={}
		n=n+data.images[1]:size(1)
		for k,v in pairs(data.images) do
			data.images[k]=net_utils.prepro(data.images[k], true, opt.gpuid>=0) --for循环，可能比较慢
		end	
		-- forward the ConvNet on images (most work happens here)
		for k,v in pairs(data.images) do
			data.raw_images[k]=data.images[k] -- raw_images存原始的(N,3,224,224)的图像tensor
			data.images[k] = protos.cnn:forward(data.images[k])
		end
		local logprobs = protos.lm:forward(data)
		--utils.logprob_equal(logprobs)
		local loss = protos.crit:forward(logprobs, data.labels)
		loss_sum = loss_sum + loss
		loss_evals = loss_evals + 1
		-- forward the model to also get generated samples for each image
		local seq = protos.lm:sample(data)
		local sents={}
		for k=1,#seq do
			local sents_t=net_utils.decode_sequence(vocab, seq[k])
			table.insert(sents,sents_t)
			print(seq[k])
			--print(torch.type(sents_t))
			print(sents_t)
		end

		for k=1, opt.batch_size do
			local story_id=data.infos[k]
			local story_txt=''
			for j=1, opt.images_per_story do
				for w in sents[k][j] do
					story_txt=story_txt .. ' ' .. w
				end
			end
			local item={}
			item.caption=story_txt
			item.image_id=story_id
			table.insert(predict,item)
		end

		if loss_evals % 10 == 0 then collectgarbage() end
		if n >= num_stories then break end -- we've used enough images
	end
	return loss_sum/loss_evals
end

local loss = eval_split(opt.split, {num_images = opt.num_stories})
print('loss: ', loss)
utils.write_json('eval/results/predict.json', predict)