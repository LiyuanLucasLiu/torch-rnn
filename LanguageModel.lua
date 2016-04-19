require 'torch'
require 'nn'

require 'VanillaRNN'
require 'LSTM'
require 'LLSTM'

local utils = require 'util.utils'


local LM, parent = torch.class('nn.LanguageModel', 'nn.Module')


function LM:__init(kwargs)
  self.batch_size = utils.get_kwarg(kwargs, 'batch_size')
  self.vocab_size = utils.get_kwarg(kwargs, 'vocab_size')
  self.model_type = utils.get_kwarg(kwargs, 'model_type')
  self.wordvec_dim = utils.get_kwarg(kwargs, 'wordvec_size')
  self.rnn_size = utils.get_kwarg(kwargs, 'rnn_size')
  self.num_layers = utils.get_kwarg(kwargs, 'num_layers')
  self.dropout = utils.get_kwarg(kwargs, 'dropout')
  self.batchnorm = utils.get_kwarg(kwargs, 'batchnorm')
  self.num_styles = utils.get_kwarg(kwargs, 'num_styles')

  local N, V, D, H = self.batch_size, self.vocab_size, self.wordvec_dim, self.rnn_size

  self.net = nn.Sequential()
  self.rnns = {}
  self.bn_view_in = {}
  self.bn_view_out = {}

  self.net:add(nn.LookupTable(V, D))
  for i = 1, self.num_layers do
    local prev_dim = H
    if i == 1 then prev_dim = D end
    local rnn
    -- if self.model_type == 'rnn' then
    --   rnn = nn.VanillaRNN(prev_dim, H)
    -- elseif self.model_type == 'lstm' then
    if (i == self.num_layers) then
      rnn = nn.LLSTM(prev_dim, H)
    else
      rnn = nn.LSTM(prev_dim, H)
    end
    -- end
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

  -- After all the RNNs run, we will have a tensor of shape (N, H);
  -- we want to apply a 1D temporal convolution to predict scores for each
  -- vocab element, giving a tensor of shape (N, T, V). Unfortunately
  -- nn.TemporalConvolution is SUPER slow, so instead we will use a pair of
  -- views (N, T, H) -> (NT, H) and (NT, V) -> (N, T, V) with a nn.Linear in
  -- between. Unfortunately N and T can change on every minibatch, so we need
  -- to set them in the forward pass.

  -- self.view1 = nn.View(1, 1, -1):setNumInputDims(3)
  -- self.view2 = nn.View(1, -1):setNumInputDims(2)

  self.net:add(nn.Linear(H, H))
  self.net:add(nn.ReLU())
  self.net:add(nn.Linear(H, H))
  self.net:add(nn.ReLU())
  self.net:add(nn.Linear(H, 1))--self.num_styles))

end


function LM:updateOutput(input)
  -- local N, T = input:size(1), input:size(2)
  -- self.view1:resetSize(N * T, -1)
  -- self.view2:resetSize(N, T, -1)

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

function LM:clearState()
  self.net:clearState()
end
