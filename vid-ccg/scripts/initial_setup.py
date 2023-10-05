import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
from zipfile import ZipFile

import nltk
import requests
from tqdm.auto import tqdm

import depccg

from . import script_utils

DEBUG = False


def main():
    nltk.download("wordnet")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("universal_tagset")
    download_spacy_command = "python3 -m spacy download en_core_web_sm"

    p = subprocess.Popen(
        download_spacy_command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    for line in p.stdout:
        print(line.decode("utf-8").rstrip())
    p.wait()
    status = p.poll()
    print("process terminate with code: %s" % status)
    download_wiki_news()


def download_wiki_news():
    link = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
    file_name = "wiki-news-300d-1M.vec"
    file_path = os.path.join("data", file_name)
    if not os.path.isfile(file_path):
        # make an HTTP request within a context manager
        with requests.get(link, stream=True) as r:
            # check header to get content length, in bytes
            total_length = int(r.headers.get("Content-Length"))

            # implement progress bar via tqdm
            with tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as raw:
                # save the output to a file
                with tempfile.NamedTemporaryFile() as tmp:
                    # with open(f"{os.path.basename(r.url)}", 'wb')as output:
                    shutil.copyfileobj(raw, tmp)
                    with ZipFile(tmp.name, "r") as zip_ref:
                        zip_ref.extract("wiki-news-300d-1M.vec", path="data")
    else:
        print(f"{file_name} already exists")


def move_dep_ccg_model(model_file):
    depccg_path = depccg.__path__[0]
    model_directory = os.path.join(depccg_path, "models")
    filename = os.path.join(model_directory, "tri_headfirst.tar.gz")
    shutil.copy(model_file, filename)
    logging.info("model copied")

    logging.info("extracting files")
    tf = tarfile.open(filename)
    tf.extractall(model_directory)
    logging.info("model extracted")


if __name__ == "__main__":
    script_utils.init_log("initial_setup", debug=DEBUG)
    main()
