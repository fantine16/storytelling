require 'hdf5'
local utils = require 'misc.utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)

	-- load the json file which contains additional information about the dataset
	print('DataLoader loading json file: ', opt.json_file)
	self.info = utils.read_json(opt.json_file)
	self.ix_to_word = self.info.ix_to_word
	self.vocab_size = utils.count_keys(self.ix_to_word)
	print('vocab size is ' .. self.vocab_size)

	-- open the hdf5 file
	print('DataLoader loading h5 file: ', opt.h5_file)
	self.h5_file = hdf5.open(opt.h5_file, 'r')

	-- extract image size from dataset
	local images_size = self.h5_file:read('/images'):dataspaceSize()
	assert(#images_size == 4, '/images should be a 4D tensor')
	assert(images_size[3] == images_size[4], 'width and height must match')
	self.num_images = images_size[1]
	self.num_channels = images_size[2]
	self.max_image_size = images_size[3]
	print(string.format('read %d images of size %dx%dx%d', self.num_images, self.num_channels, self.max_image_size, self.max_image_size))
	

	-- load in the sequence data
	local seq_size = self.h5_file:read('/labels'):dataspaceSize()
	self.seq_length = seq_size[2]
	print('max sequence length in data is ' .. self.seq_length)
	-- load the pointers in full to RAM (should be small enough)
	self.label_start_ix = self.h5_file:read('/label_start_ix'):all()
	-- separate out indexes for each of the provided splits
	self.split_ix = {}
	self.iterators = {}
	for i,term in pairs(self.info.story) do
		local split = term.split
		if not self.split_ix[split] then
			-- initialize new split
			self.split_ix[split] = {}
			self.iterators[split] = 1
		end
		table.insert(self.split_ix[split], i)
	end
	for k,v in pairs(self.split_ix) do
		print(string.format('assigned %d stories to split %s', #v, k))
	end

end

function DataLoader:resetIterator(split)
  self.iterators[split] = 1
end

function DataLoader:getVocabSize()
	return self.vocab_size
end

function DataLoader:getVocab()
	return self.ix_to_word
end

function DataLoader:getSeqLength()
	return self.seq_length
end

function DataLoader:getBatch(opt)
	local split = utils.getopt(opt, 'split') -- lets require that user passes this in, for safety
	local batch_size = utils.getopt(opt, 'batch_size', 5) -- how many stories get returned at one time (to go through CNN)
	local images_per_story = utils.getopt(opt, 'images_per_story', 5) -- number of images to return per story

	local split_ix = self.split_ix[split]
	assert(split_ix, 'split ' .. split .. ' not found.')
	local max_index = #split_ix
	local wrapped = false

	-- pick an index of the datapoint to load next
	local story_batch={}

	for i=1,batch_size do
		local onestory={}
		local imgs_per_story=torch.ByteTensor(images_per_story, 3, 256, 256)
		local labels_per_story=torch.LongTensor(images_per_story, self.seq_length)
		local ri = self.iterators[split] -- get next index from iterator
		local ri_next = ri + 1
		if ri_next > max_index then ri_next = 1; wrapped = true end -- wrap back around
		self.iterators[split] = ri_next
		ix = split_ix[ri]
		assert(ix ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. ri)

		--print('DataLoader 93: ri=' .. ri .. ';ri_next=' .. ri_next .. ';max_index=' .. max_index)
		--[[
		for j =1, images_per_story do
			local k = images_per_story*(ix-1) + j
			--print('DataLoader 95: split=' .. split .. ';i=' .. i .. ';j=' .. j .. ';k=' .. k .. ';ix= ' .. ix)
			imgs_per_story[j]=self.h5_file:read('/images'):partial({k, k},{1,self.num_channels},{1,self.max_image_size},{1,self.max_image_size})
			labels_per_story[j]=self.h5_file:read('/labels'):partial({k, k}, {1,self.seq_length})
		end]]

		imgs_per_story=self.h5_file:read('/images'):partial({images_per_story*(ix-1) + 1, images_per_story*ix},{1,self.num_channels},{1,self.max_image_size},{1,self.max_image_size})
		labels_per_story=self.h5_file:read('/labels'):partial({images_per_story*(ix-1) + 1, images_per_story*ix}, {1,self.seq_length})



		onestory.images=imgs_per_story
		onestory.labels=labels_per_story:contiguous() --:transpose(1,2) :contiguous()-- note: make label sequences go down as columns
		table.insert(story_batch,onestory)
	end

	local imgs={} -- imgs的第t个元素是 story 的第t个图像， 每个元素是 images tensor，(batch_size*3*224*224)
	local labels={} -- labels的第t个元素是story的第t个标注， 每个元素是 tensor，(batch_size*seq_length)
	local batch_size=#story_batch
	--print('batch_size=' .. batch_size)
	for t=1,images_per_story do
		local im = torch.ByteTensor(batch_size,3,256,256)
		local la =torch.LongTensor(batch_size,self.seq_length)
		for k=1,batch_size do
			--print('t=' .. t .. ';k=' .. k)
			im[k]=story_batch[k].images[t]
			la[k]=story_batch[k].labels[t]
		end
		table.insert(imgs,im)
		table.insert(labels,la)
	end
	local data={}
	data.images=imgs
	data.labels=labels
	return data

end
