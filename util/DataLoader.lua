require 'torch'
require 'hdf5'

local utils = require 'util.utils'

local DataLoader = torch.class('DataLoader')


function DataLoader:__init(kwargs, num4padding)
  local h5_file = utils.get_kwarg(kwargs, 'input_h5')
  self.batch_size = utils.get_kwarg(kwargs, 'batch_size')
  -- self.seq_length = utils.get_kwarg(kwargs, 'seq_length')
  -- local N, T = self.batch_size, self.seq_length

  -- Just slurp all the data into memory
  local f = hdf5.open(h5_file, 'r')
  self.num4padding = num4padding
  self.x_splits = {train = {}, test = {}, val = {}}
  self.y_splits = {train = {}, test = {}, val = {}}
  self.split_sizes = {train = {}, test = {}, val = {}}

  for idx=1,num4padding do
    local revidx = num4padding-idx+1
    local vx = f:read('/x_train'..revidx):all()
    local vy = f:read('/y_train'..revidx):all()
    local tmpsize = vx:size(1)
    tmpsize = tmpsize - tmpsize % self.batch_size
    self.x_splits['train'][idx] = vx[{{1, tmpsize}, {}}]:view(tmpsize/self.batch_size, self.batch_size, -1):clone() 
    self.y_splits['train'][idx] = vy[{{1, tmpsize}}]:view(tmpsize/self.batch_size, self.batch_size):clone()
    self.split_sizes['train'][idx] = tmpsize/self.batch_size
    
    vx = f:read('/x_val'..idx):all()
    vy = f:read('/y_val'..idx):all()
    tmpsize = vx:size(1)
    tmpsize = tmpsize - tmpsize%self.batch_size
    self.x_splits['val'][idx] = vx[{{1, tmpsize}, {}}]:view(tmpsize/self.batch_size, self.batch_size, -1):clone()
    self.y_splits['val'][idx] = vy[{{1, tmpsize}}]:view(tmpsize/self.batch_size, self.batch_size, 1):clone()
    self.split_sizes['val'][idx] = tmpsize/self.batch_size
    
    vx = f:read('/x_test'..idx):all()
    vy = f:read('/y_test'..idx):all()
    tmpsize = vx:size(1)
    tmpsize = tmpsize - tmpsize%self.batch_size
    self.x_splits['test'][idx] = vx[{{1, tmpsize}, {}}]:view(tmpsize/self.batch_size, self.batch_size, -1):clone()
    self.y_splits['test'][idx] = vy[{{1, tmpsize}}]:view(tmpsize/self.batch_size, self.batch_size, 1):clone()
    self.split_sizes['test'][idx] = tmpsize/self.batch_size

  end

  self.split_idx1 = {train=1, val=1, test=1}
  self.split_idx2 = {train=1, val=1, test=1}
end


function DataLoader:nextBatch(split)
  local idx1 = self.split_idx1[split]
  local idx2 = self.split_idx2[split]
  local x = self.x_splits[split][idx1][idx2]
  local y = self.y_splits[split][idx1][idx2]
  if idx2 == self.split_sizes[idx2] then
    self.split_idx2[split] = 1
    if idx1 == self.num4padding then
      self.split_idx1[split] = 1
    else
      self.split_idx1[split] = idx1 + 1
    end
  else
    self.split_idx2[split] = idx2 + 1
  end
  return x, y
end

