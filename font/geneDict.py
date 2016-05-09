import json
import PIL
import numpy as np
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

font = ImageFont.truetype("MSYH.TTC", 20, encoding = "unic")

with open('CharTable.json', 'rb') as f:
	ct = json.loads(f.read())

max_size_1 = 0
max_size_0 = 0
for tup in ct:
	tmpsize = font.getsize(tup)
	max_size_0 = max(max_size_0, tmpsize[0])
	max_size_1 = max(max_size_1, tmpsize[1])

max_size_1 = max_size_1 + 1
max_size_0 = max_size_0 + 1
word = np.empty((len(ct), max_size_1, max_size_0), dtype = 'uint8')
for idx in range(0, len(ct)):
	img = Image.new("L", (max_size_0, max_size_1), "white")
	draw = ImageDraw.Draw(img)
	size = font.getsize(ct[idx]) 
	
	draw.text((int(max_size_0-size[0])/2, int(max_size_1-size[1])/2), ct[idx], 0, font=font)
	
	word[idx] = np.array(img)

with open('dict.np', 'wb') as f:
	np.save(f, word)

