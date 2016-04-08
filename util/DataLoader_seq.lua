require 'torch'
require 'hdf5'

local utils = require 'util.utils'

local DataLoader_seq = torch.class('DataLoader_seq')


function DataLoader_seq:__init(kwargs)
  local h5_file = utils.get_kwarg(kwargs, 'input_h5')
  self.batch_size = utils.get_kwarg(kwargs, 'batch_size')
  self.seq_length = utils.get_kwarg(kwargs, 'seq_length')
  local N, T = self.batch_size, self.seq_length

  -- Just slurp all the data into memory
  local splits = {}
  local splits_y = {}
  local f = hdf5.open(h5_file, 'r')
  self.max_length = f:read('/max_length'):all()

  splits.train = f:read('/train'):all()
  splits.val = f:read('/val'):all()
  splits.test = f:read('/test'):all()

  splits_y.train= f:read('/train_y'):all()
  splits_y.val= f:read('/val_y'):all()
  splits_y.test= f:read('/test_y'):all()

  self.x_splits = {}
  self.y_splits = {}
  self.split_sizes = {}
  for split, v in pairs(splits) do
    local num = v:nElement()
    local extra = num % (N * T)

    -- Chop out the extra bits at the end to make it evenly divide
    local vx = v[{{1, num - extra}}]:clone()
    self.x_splits[split] = vx
    self.split_sizes[split] = vx:size(1)
  end

  for split, v in pairs(splits) do
    local num = v:nElement()
    local extra = num % (N * T)

    -- Chop out the extra bits at the end to make it evenly divide
    local vy = v[{{2, num - extra + 1}}]:clone()
    self.y_splits[split] = vy
  end

  self.split_idxs = {train=1, val=1, test=1}
end


function DataLoader_seq:nextBatch(split)
  local idx = self.split_idxs[split]
  assert(idx, 'invalid split ' .. split)
  local x = self.x_splits[split][idx]
  local y = self.y_splits[split][idx]
  if idx == self.split_sizes[split] then
    self.split_idxs[split] = 1
  else
    self.split_idxs[split] = idx + 1
  end
  return x, y
end

