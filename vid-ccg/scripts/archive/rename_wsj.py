import os

dir_path = "/Users/rubensteendam/nltk_data/corpora/ptb/WSJ"

res = []
for root, dirs, files in os.walk(dir_path, topdown=False):
    for name in files:
        if name.endswith(".mrg"):
            os.rename((os.path.join(root, name)), (os.path.join(root, name.upper())))
