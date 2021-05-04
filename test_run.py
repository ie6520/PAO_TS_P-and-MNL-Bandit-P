from algorithm import run
import os
import json

name = input("name:")
filename = name+".json"
file = os.path.join("./config", filename)
with open(file, "r") as f:
    datas = json.load(f)["setting"]

for data in datas:
    run(*data)