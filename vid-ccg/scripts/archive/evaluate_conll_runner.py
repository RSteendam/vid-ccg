import sys
from os import listdir, system
from os.path import isdir, join

output_dir = sys.argv[1]
gold_file = sys.argv[2]

paths = [join(output_dir, f) for f in listdir(output_dir) if isdir(join(output_dir, f))]
for directory in paths:
    file = join(directory, "Test.0.1.JSON.gz")
    system(f"python3 -m scripts.evaluate_conll {file} {gold_file}")
