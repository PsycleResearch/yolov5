import json

with open('./datas/labels.json', 'r') as f:
    datas = json.load(f)

temp = {}
for key,values in datas.items():

    if not None in values:
        t = []
        for value in values:
            t.append(value)
        temp[key] = t

with open('./datas/temp.json', 'w') as f:
    json.dump(temp, f)