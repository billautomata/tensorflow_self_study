import json
from pprint import pprint
import numpy as np

with open('values.json') as data_file:
    data = json.load(data_file)

print(len(data[0]))

w = 42034
h = 257

vals = np.zeros((w,h), dtype=np.int32)

print(vals.shape)

for i, v in enumerate(data):
    vals.itemset((v[0], v[1]), v[2])

print(range(10))

n = 10000

for x in range(n,n+10):
    print(vals[n])
