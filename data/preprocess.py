import tweet as tt

data = tt.tweet()
data.process(4, 0.1, 0.1)
data.saveas('4block.js')
