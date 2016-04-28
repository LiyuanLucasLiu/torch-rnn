require 'torch'
require 'nn'

require 'VanillaRNN'
require 'LSTM'

local utils = require 'util.utils'


local LM, parent = torch.class('nn.LanguageModel', 'nn.Module')


function LM:__init(kwargs)
  self.idx_to_token = utils.get_kwarg(kwargs, 'idx_to_token')
  self.token_to_idx = {}
  self.vocab_size = 0
  for idx, token in pairs(self.idx_to_token) do
    self.token_to_idx[token] = idx
    self.vocab_size = self.vocab_size + 1
  end

  self.model_type = utils.get_kwarg(kwargs, 'model_type')
  self.wordvec_dim = utils.get_kwarg(kwargs, 'wordvec_size')
  self.rnn_size = utils.get_kwarg(kwargs, 'rnn_size')
  self.num_layers = utils.get_kwarg(kwargs, 'num_layers')
  self.dropout = utils.get_kwarg(kwargs, 'dropout')
  self.batchnorm = utils.get_kwarg(kwargs, 'batchnorm')

  self.channelSize = utils.get_kwarg(kwargs, 'channelSize')
  self.dropProb = utils.get_kwarg(kwargs, 'dropoutProb')
  self.kernelSize = utils.get_kwarg(kwargs, 'kernelSize')
  self.kernelStride = utils.get_kwarg(kwargs, 'kernelStride')
  self.padding = utils.get_kwarg(kwargs, 'padding')
  self.batchNorm = utils.get_kwarg(kwargs, 'batchnorm')
  self.poolSize = utils.get_kwarg(kwargs, 'poolSize')
  self.poolStride = utils.get_kwarg(kwargs, 'poolStride')
  self.activation = utils.get_kwarg(kwargs, 'activation')

  local V, D, H = self.vocab_size, self.wordvec_dim, self.rnn_size

  self.net = nn.Sequential()
  self.rnns = {}

  inputSize = 1
  depth = 1
  for i = 1, #self.channelSize do 
    if self.dropProb[depth] > 0 then
      self.net:add(nn.SpatialDropout(self.dropProb[depth]))
    end
    self.net:add(nn.SpatialConvolution(
      inputSize, self.channelSize[i],
      self.kernelSize[i], self.kernelSize[i],
      self.kernelStride[i], self.kernelStride[i],
      self.padding and math.floor(self.kernelSize[i]/2) or 0
    ))
    if self.batchNorm then
      self.net:add(nn.SpatialBatchNormalization(self.channelSize))
    end
    self.net:add(nn[self.activation]())
    if self.poolSize[i] and self.poolSize[i] > 0 then
      self.net:add(nn.SpatialMaxPooling(
        self.poolSize[i], self.poolSize[i],
        self.poolStride[i] or self.poolSize[i],
        self.poolStride[i] or self.poolsize[i]
      ))
    end
    inputSize = self.channelSize[i]
    depth = depth + 1
  end
  self.view3 = nn.View(-1):setNumInputDims(2)
  self.view4 = nn.View(1, 1, -1):setNumInputDims(2)

  self.net:add(self.view3)
  self.net:add(nn.Linear())
  self.net:add(self.view4)

  for i = 1, self.num_layers do
    local prev_dim = H
    if i == 1 then prev_dim = D end
    local rnn
    if self.model_type == 'rnn' then
      rnn = nn.VanillaRNN(prev_dim, H)
    elseif self.model_type == 'lstm' then
      rnn = nn.LSTM(prev_dim, H)
    end
    rnn.remember_states = true
    table.insert(self.rnns, rnn)
    self.net:add(rnn)
    if self.batchnorm == 1 then
      local view_in = nn.View(1, 1, -1):setNumInputDims(3)
      table.insert(self.bn_view_in, view_in)
      self.net:add(view_in)
      self.net:add(nn.BatchNormalization(H))
      local view_out = nn.View(1, -1):setNumInputDims(2)
      table.insert(self.bn_view_out, view_out)
      self.net:add(view_out)
    end
    if self.dropout > 0 then
      self.net:add(nn.Dropout(self.dropout))
    end
  end

  -- After all the RNNs run, we will have a tensor of shape (N, T, H);
  -- we want to apply a 1D temporal convolution to predict scores for each
  -- vocab element, giving a tensor of shape (N, T, V). Unfortunately
  -- nn.TemporalConvolution is SUPER slow, so instead we will use a pair of
  -- views (N, T, H) -> (NT, H) and (NT, V) -> (N, T, V) with a nn.Linear in
  -- between. Unfortunately N and T can change on every minibatch, so we need
  -- to set them in the forward pass.
  self.view1 = nn.View(1, 1, -1):setNumInputDims(3)
  self.view2 = nn.View(1, -1):setNumInputDims(2)

  self.net:add(self.view1)
  self.net:add(nn.Linear(H, V))
  self.net:add(self.view2)
end


function LM:updateOutput(input)
  local N, T = input:size(1), input:size(2)
  self.view1:resetSize(N * T, -1)
  self.view2:resetSize(N, T, -1)

  for _, view_in in ipairs(self.bn_view_in) do
    view_in:resetSize(N * T, -1)
  end
  for _, view_out in ipairs(self.bn_view_out) do
    view_out:resetSize(N, T, -1)
  end

  return self.net:forward(input)
end


function LM:backward(input, gradOutput, scale)
  return self.net:backward(input, gradOutput, scale)
end


function LM:parameters()
  return self.net:parameters()
end


function LM:training()
  self.net:training()
  parent.training(self)
end


function LM:evaluate()
  self.net:evaluate()
  parent.evaluate(self)
end


function LM:resetStates()
  for i, rnn in ipairs(self.rnns) do
    rnn:resetStates()
  end
end


function LM:encode_string(s)
  local encoded = torch.LongTensor(#s)
  for i = 1, #s do
    local token = s:sub(i, i)
    local idx = self.token_to_idx[token]
    assert(idx ~= nil, 'Got invalid idx')
    encoded[i] = idx
  end
  return encoded
end


function LM:decode_string(encoded)
  assert(torch.isTensor(encoded) and encoded:dim() == 1)
  local s = ''
  for i = 1, encoded:size(1) do
    local idx = encoded[i]
    local token = self.idx_to_token[idx]
    s = s .. token
  end
  return s
end


--[[
Sample from the language model. Note that this will reset the states of the
underlying RNNs.

Inputs:
- init: String of length T0
- max_length: Number of characters to sample

Returns:
- sampled: (1, max_length) array of integers, where the first part is init.
--]]
function LM:sample(kwargs)
  local T = utils.get_kwarg(kwargs, 'length', 100)
  local start_text = utils.get_kwarg(kwargs, 'start_text', '')
  local verbose = utils.get_kwarg(kwargs, 'verbose', 0)
  local sample = utils.get_kwarg(kwargs, 'sample', 1)
  local temperature = utils.get_kwarg(kwargs, 'temperature', 1)

  local sampled = torch.LongTensor(1, T)
  self:resetStates()

  local scores, first_t
  if #start_text > 0 then
    if verbose > 0 then
      print('Seeding with: "' .. start_text .. '"')
    end
    local x = self:encode_string(start_text):view(1, -1)
    local T0 = x:size(2)
    sampled[{{}, {1, T0}}]:copy(x)
    scores = self:forward(x)[{{}, {T0, T0}}]
    first_t = T0 + 1
  else
    if verbose > 0 then
      print('Seeding with uniform probabilities')
    end
    local w = self.net:get(1).weight
    scores = w.new(1, 1, self.vocab_size):fill(1)
    first_t = 1
  end
  
  local _, next_char = nil, nil
  for t = first_t, T do
    if sample == 0 then
      _, next_char = scores:max(3)
      next_char = next_char[{{}, {}, 1}]
    else
       local probs = torch.div(scores, temperature):double():exp():squeeze()
       probs:div(torch.sum(probs))
       next_char = torch.multinomial(probs, 1):view(1, 1)
    end
    sampled[{{}, {t, t}}]:copy(next_char)
    scores = self:forward(next_char)
  end

  self:resetStates()
  return self:decode_string(sampled[1])
end

--[[
Sample from the language model. Note that this will reset the states of the
underlying RNNs.

sample with beam search, most of code are borrowed from http://github.com/pender/char-rnn
--]]
function LM:sample_beam(opt)
  local T = utils.get_kwarg(opt, 'length', 100)
  local start_text = utils.get_kwarg(opt, 'start_text', '')
  local verbose = utils.get_kwarg(opt, 'verbose', 0)
  local sample = utils.get_kwarg(opt, 'sample', 1)
  local temperature = utils.get_kwarg(opt, 'temperature', 1)

  local sampled = torch.LongTensor(1, T)
  self:resetStates()

  local prediction, first_t
  if #start_text > 0 then
    if verbose > 0 then
      print('Seeding with: "' .. start_text .. '"')
    end
    local x = self:encode_string(start_text):view(1, -1)
    local T0 = x:size(2)
    sampled[{{}, {1, T0}}]:copy(x)
    prediction = self:forward(x)[{{}, {T0, T0}}]
    first_t = T0 + 1
  else
    if verbose > 0 then
      print('Seeding with uniform probabilities')
    end
    local w = self.net:get(1).weight
    prediction = w.new(1, 1, self.vocab_size):fill(1)
    first_t = 1
  end
  
  local index_sampled = first_t

  -- Print a given node of the string tree to the screen.
  -- Strings are stored as a branching linked list of characters.
  function printNode(node)
      local currentStringTail = node
      local backwardString = {}
      while currentStringTail do
          backwardString[#backwardString + 1] = currentStringTail.value
          currentStringTail = currentStringTail.parent
      end
      for i = #backwardString, 1, -1 do
          sampled[{{}, {index_sampled, index_sampled}}] = backwardString[i]
	        index_sampled = index_sampled + 1
          -- io.write(ivocab[backwardString[i]])
      end
  end

  function dprint(str)
    if opt.debug == 1 then print(str) end
  end

  -- Print the portion of the string on which the beam has reached consensus so far.
  function printFinalizedCharacters(stringTails)
      if (#stringTails == 1) then
          if (stringTails[1].parent ~= nil) then
              -- Automatically print if there's only one string tail (i.e. opt.beam == 1).
              printNode(stringTails[1].parent)
              stringTails[1].parent = nil
          end
      else
          local tailIterators = {}
          local count = 0
          for k, v in ipairs(stringTails) do
              if (v.parent ~= nil and tailIterators[v.parent] == nil) then
                  tailIterators[v.parent] = true
                  count = count + 1
              end
          end
          local lastTail;
          -- Trace the string heads backward until they form a common trunk.
          while count > 1 do
              count = 0
              local newTailIterators = {}
              for stringTail, _ in pairs(tailIterators) do
                  if (stringTail.parent ~= nil and newTailIterators[stringTail.parent] == nil) then
                      newTailIterators[stringTail.parent] = true
                      count = count + 1
                      lastTail = stringTail.parent
                  end
              end
              tailIterators = newTailIterators
          end
          -- Print that trunk, and then chop it off.
          if lastTail ~= nil and lastTail.parent ~= nil then
              printNode(lastTail.parent) -- Print through here.
              lastTail.parent = nil -- Cut the trunk off.
          end
      end
  end

  -- Function to boost probabilities (multiplicatively and equally) when they're getting too low.
  -- Otherwise they eventually all round down to zero.
  -- It is important that the probabilities maintain their relative proportions, not that they be correct absolutely.
  function boostProbabilities(prob_list)
      local max = 0
      for probIndex,currentProb in ipairs(prob_list) do
          if currentProb > max then
              max = currentProb
          end
      end
      while max < 0.0001 do
          for i = 1,#prob_list do
              prob_list[i] = prob_list[i] * 1000
          end
          max = max * 1000
      end
  end

  -- start sampling/argmaxing/beam searching
  local states = {} -- stores the best opt.beam activation states
  local cum_probs = {} -- stores the corresponding cumulative probabilities (periodically boosted with boostProbabilities)
  local stringTails = {} -- stores the corresponding string tails generated by the states so far
  -- initially populate states table with the net
  states[#states + 1] = self:clone()
  current_state = nil
  cum_probs[#cum_probs + 1] = 1

  local timer = torch.Timer()

  for outputIndex= first_t, opt.length do
      dprint("\nPicking character #" .. outputIndex)
      -- local newStateIndices = {}
      local newCumProbs = {}
      local newStringTails = {}
      local newStates = {}
      for stateIndex,stateContent in ipairs(states) do
          if (outputIndex > first_t or stateIndex > 1) then -- The state was already loaded above if this is the first character.
              -- Pull the previous character.
              prev_char = torch.Tensor{stringTails[stateIndex].value}:view(1,1)
              -- -- Forward the latest character and extract the probabilities that result.
              -- if opt.debug == 1 then print("state #" .. stateIndex .. ", forwarding character '" .. ivocab[prev_char[1]] .. "'") end
              -- local lst = protos.rnn:forward{prev_char, unpack(stateContent)}
              -- local newStateContent = {}
              -- for i=1,state_size do table.insert(newStateContent, lst[i]:clone()) end -- clone to avoid entangling with other entries in state[].
              -- states[stateIndex] = newStateContent -- Save the modified state back to the state table.
              -- prediction = lst[#lst] -- log probabilities
              prediction = stateContent:forward(prev_char)
          end
          -- Get the probabilities of each character at the current state.
          prediction:div(opt.temperature) -- scale by temperature
          local probs = torch.exp(prediction):squeeze()
          probs:div(torch.sum(probs)) -- renormalize so probs sum to one and are actual probabilities
          -- Populate currentBestChars with the top opt.beam characters, in order of likelihood.
          local currentBestChars = {}
          local probsCopy = probs:clone() -- Clone probabilities tensor so we can zero out items as we draw them.
          for candidate=1, opt.beam do
              if (probsCopy:max() <= 0) then
                  break
              end
              local char_index
              if opt.sample == 0 then
                  -- Pull the highest-probability character index.
                  local _, prev_char = probsCopy:max(1)
                  char_index = prev_char[1]
              else
                  -- Sample a character index.
                  prev_char = torch.multinomial(probsCopy:float(), 1):resize(1):float()
                  char_index = prev_char[1]
              end
              if opt.debug == 1 then print("state #" .. stateIndex .. ", option #" .. candidate .. ": "
                          .. char_index .. " ('" .. ivocab[char_index] .. "'); prob: " .. probs[char_index]) end
              probsCopy[char_index] = 0 -- Zero out that index so we don't pull it again.
              currentBestChars[#currentBestChars + 1] = char_index -- Add it to the list of best characters at this node.
          end
          -- For each of the characters in currentBestChars, check its probability and keep a rolling
          -- record of the best states in newStateIndices. How many states to keep is defined by opt.beam.
          for _, char_index in ipairs(currentBestChars) do
              local cumProb = probs[char_index] * cum_probs[stateIndex] -- Cumulative probability of this character choice
              if cumProb > 0 then -- If cumProb is equal to zero, this is a dead end.
                  local insertionPoint = -1
                  if #newStates < opt.beam then
                       -- If newStates has fewer entries than the beam width, we automatically qualify.
                      insertionPoint = #newStates + 1
                  else
                      local probsTensor = torch.Tensor(newCumProbs);
                      if opt.beamsample == 0 then
                          -- Find the lowest cumulative probability and its index in newCumProbs.
                          local _, min = probsTensor:min(1)
                          local minIndex = min[1]
                          if (probsTensor[minIndex] <= cumProb) then
                              insertionPoint = minIndex
                          end
                      else
                          -- Sample the beam states and randomly draw a low one.
                          probsTensor:div(opt.temperature) -- scale by temperature
                          probsTensor:div(torch.sum(probsTensor)) -- renormalize so probs sum to one
                          -- Since we want a low one, we first have to invert the probabilities in the tensor.
                          local min = probsTensor:min(1); local minValue = min[1]
                          local max = probsTensor:max(1); local maxValue = max[1]
                          probsTensor = -probsTensor + maxValue + minValue

                          local min = probsTensor:min(1); local max = probsTensor:max(1)
                          if (max[1] <= 0 or min[1] < 0) then
                              print("Error with probabilities tensor: \n", probsTensor)
                          end
                          local index_ = torch.multinomial(probsTensor, 1):resize(1):float()
                          local index = index_[1]
                          if (newCumProbs[index] <= cumProb) then
                              insertionPoint = index
                          end
                      end
                  end
                  if insertionPoint > 0 then
                      newStates[insertionPoint] = stateContent:clone()
                      -- newStateIndices[insertionPoint] = stateIndex;
                      newCumProbs[insertionPoint] = cumProb
                      local newStringTail = {parent = stringTails[stateIndex], value = char_index}
                      newStringTails[insertionPoint] = newStringTail
                  end
              end
          end
      end

      -- Replace the old states with the new.
      -- local newStates = {}
      -- for iterator, newIndex in ipairs(newStateIndices) do
      --     dprint("Entry " .. iterator .. ": Cloning from state index ".. newIndex)
      --     newStates[iterator] = states[newIndex]
      -- end

      states = newStates;
      cum_probs = newCumProbs;
      stringTails = newStringTails;
      if (opt.debug == 1) then
          for stateIndex=1,#stringTails do
              dprint(string.format("          State #%i, prob %.3e:", stateIndex, cum_probs[stateIndex]))
              printNode(stringTails[stateIndex])
              io.write('\n'); io.flush()
          end
      end
      -- Boost the probabilities if they're getting too low; all that matters is relative probabilities,
      -- and with long enough text they could otherwise exceed floating point accuracy and zero out completely.
      boostProbabilities(cum_probs)
      -- Print however many characters the beam has reached consensus about,
      -- but do it less frequently with a wider beam to avoid churning needlessly.
      if outputIndex % opt.beam == 0 then printFinalizedCharacters(stringTails) end
      -- Periodically take out the trash.
      if outputIndex % 10 == 0 then collectgarbage() end
  end

  -- Pick the winning state.
  local max = 0
  local winningIndex = 0
  for probIndex,currentProb in ipairs(cum_probs) do
      if currentProb > max then
          max = currentProb
          winningIndex = probIndex
      end
  end
  -- Transcribe the winning string.
  printNode(stringTails[winningIndex])


  self:resetStates()
  return self:decode_string(sampled[1])
end

function LM:clearState()
  self.net:clearState()
end
