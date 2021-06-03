import json

# with open('./datas/labels.json', 'r') as f:
#     datas = json.load(f)
#
# temp = {}
# for key,values in datas.items():
#
#     if not None in values:
#         t = []
#         for value in values:
#             t.append(value)
#         temp[key] = t
#
# with open('./datas/temp.json', 'w') as f:
#     json.dump(temp, f)

with open('./datas/temp.json', 'r') as f:
    f = json.load(f)

training_set = {}
validation_set = {}

key = list(f.keys())
split_idx = int(0.8 * len(key))
train_key, val_key = key[:split_idx], key[split_idx:]

for key in train_key:
    training_set[key] = f[key]

for key in val_key:
    validation_set[key] = f[key]

with open('./datas/training_set.json', 'w') as f:
    json.dump(training_set, f)

with open('./datas/validation_set.json', 'w') as f:
    json.dump(validation_set, f)

