import os
import json
import argparse
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--key", help="dict key")
parser.add_argument("--val", help="dict value")
args = parser.parse_args()
key = args.key
val = args.val

storage_path = os.path.join(tempfile.gettempdir(), 'storage.data')

if os.path.exists(storage_path):
    with open(storage_path, 'r') as f:
        data = json.loads((f.read()))
else:
    with open(storage_path, 'w') as f:
        f.write('{}')
        data = {}
    
if key and val:
    if key in data:
        data[key].append(val)
    else:
        data[key] = [val]
    with open(storage_path, 'w') as f:
        json.dump(data, f)
else:
    if key in data:
        print(*data[key], sep = ", ")
    else:
        print(None)
