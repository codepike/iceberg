import os
import numpy as np
import json

probs = []
with open("test.csv") as f:
    for line in f:
        prob = float(line)
        probs.append(prob)

rawdata = open('./data/test.json').read()
data = json.loads(rawdata)

i=0
print("prob len", len(probs))
print("len", len(data))

with open("batch_result.csv", 'w') as output:
    output.write("id,is_iceberg\n")
    for item in data:
        output.write("{},{}\n".format(item["id"], probs[i]))
        i += 1