require 'torch'
require 'nn'

require 'LanguageModel'


local tests = {}
local tester = torch.Tester()


local function check_dims(x, dims)
  tester:assert(x:dim() == #dims)
  for i, d in ipairs(dims) do
    tester:assert(x:size(i) == d)
  end
end


-- Just a smoke test to make sure model can run forward / backward
function tests.simpleTest()
  local N, T, D, H, V = 2, 3, 4, 5, 6
  local idx_to_token = {[1]='a', [2]='b', [3]='c', [4]='d', [5]='e', [6]='f'}
  local LM = nn.LanguageModel{
    idx_to_token=idx_to_token,
    cell_type='rnn',
    wordvec_dim=D,
    hidden_dim=H,
    num_layers=6,
  }
  local crit = nn.CrossEntropyCriterion()
  local params, grad_params = LM:getParameters()

  local x = torch.Tensor(N, T):random(V)
  local y = torch.Tensor(N, T):random(V)
  local scores = LM:forward(x)
  check_dims(scores, {N, T, V})
  local scores_view = scores:view(N * T, V)
  local y_view = y:view(N * T)
  local loss = crit:forward(scores_view, y_view)
  local dscores = crit:backward(scores_view, y_view):view(N, T, V)
  LM:backward(x, dscores)
end


function tests.sampleTest()
  local N, T, D, H, V = 2, 3, 4, 5, 6
  local idx_to_token = {[1]='a', [2]='b', [3]='c', [4]='d', [5]='e', [6]='f'}
  local LM = nn.LanguageModel{
    idx_to_token=idx_to_token,
    cell_type='rnn',
    wordvec_dim=D,
    hidden_dim=H,
    num_layers=6,
  }
  
  local TT = 10
  local init = torch.LongTensor{{2, 3, 6}}
  local sampled = LM:sample(init, TT)
  check_dims(sampled, {1, TT})
end


function tests.encodeDecodeTest()
  local idx_to_token = {
    [1]='a', [2]='b', [3]='c', [4]='d',
    [5]='e', [6]='f', [7]='g', [8]=' ',
  }
  local N, T, D, H, V = 2, 3, 4, 5, 7
  local LM = nn.LanguageModel{
    idx_to_token=idx_to_token,
    cell_type='rnn',
    wordvec_dim=D,
    hidden_dim=H,
    num_layers=6,
  }

  local s = 'a bad feed'
  local encoded = LM:encode_string(s)
  local expected_encoded = torch.LongTensor{1, 8, 2, 1, 4, 8, 6, 5, 5, 4}
  tester:assert(torch.all(torch.eq(encoded, expected_encoded)))

  local s2 = LM:decode_string(encoded)
  tester:assert(s == s2)
end

tester:add(tests)
tester:run()
