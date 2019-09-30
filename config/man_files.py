import os
import json

files = [filename for filename in os.listdir('.') if filename.startswith("scenario_13_")]

for f in files:
    index = f.split("_")[2]
    with open(f) as jfile:
        config = json.load(jfile)

    config["seed"] = str(index.split(".")[0])
    config['windows_size'] = str(1)
    wfile = "scenario_14_" + index
    
    with open(wfile, "w") as write_file:
       json.dump(config, write_file)