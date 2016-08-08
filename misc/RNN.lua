local RNN = {}

function RNN.rnn(input_size, output_size, rnn_size, stacked_num_layers ,HH_num_layers,HO_num_layers, dropout)
	
	-- there are n+1 inputs (hiddens on each layer and x)
	local inputs = {}
	table.insert(inputs, nn.Identity()()) -- x
	for L = 1,stacked_num_layers do
		table.insert(inputs, nn.Identity()()) -- prev_h[L]

	end

	local x, input_size_L
	local outputs = {}
	local HH_outputs = {}

	for L = 1, stacked_num_layers+HH_num_layers do
		if L <= stacked_num_layers then
			local prev_h = inputs[L+1]
			if L == 1 then
				x = inputs[1]
				input_size_L = input_size
			else
				x = outputs[(L-1)]
				if dropout > 0 then x = nn.Dropout(dropout)(x) end
				input_size_L = rnn_size
			end
			local i2h = nn.Linear(input_size_L, rnn_size)(x)
			local h2h = nn.Linear(rnn_size, rnn_size)(prev_h)
			local next_h = nn.Tanh()(nn.CAddTable(){i2h, h2h})

			table.insert(outputs, next_h)
		else
			x = outputs[(L-1)]
			if dropout > 0 then x = nn.Dropout(dropout)(x) end
			input_size_L = rnn_size
			local next_h = nn.Sigmoid()(nn.Linear(input_size_L, rnn_size)(x))
			table.insert(outputs, next_h)

		end		
	end

	local top_h = outputs[#outputs]
	for L = 1, HO_num_layers  do
		local proj = n.Linear(rnn_size, output_size)(top_h)
		top_h = proj
	end

	if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
	local proj = nn.Linear(rnn_size, output_size)(top_h)
	local logsoft = nn.LogSoftMax()(proj)

	table.insert(outputs, logsoft)
	local new_outputs = {}
	for i = #outputs - stacked_num_layers -1, #outputs do
		table.insert(new_outputs, outputs[i])
	end
	return nn.gModule(inputs, new_outputs)

end

return RNN
