import sys
from os import listdir, system
from os.path import isfile, join

experiment_dir = sys.argv[1]

paths = [
    join(experiment_dir, f)
    for f in listdir(experiment_dir)
    if isfile(join(experiment_dir, f))
]

for file in paths:
    print(f"running: {file}")
    system(f"python3 -m scripts.induce_grammar {file}")
