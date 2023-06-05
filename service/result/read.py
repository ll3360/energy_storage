import json
from pprint import pprint

with open('./1_20221205_20221207.json', 'r+') as f:
	data = json.load(f)
	pprint(data)
	print(len(data["data"]))