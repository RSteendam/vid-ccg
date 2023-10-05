import os
import sys
import time

import tqdm

FORBIDDEN_TAGS = ["''", "-NONE-", "``"]
CCG_PARG = "output/wsj/ccg_parg_gold.parg"
CCG_AUTO = "output/wsj/ccg_auto_gold.auto"


def main():
    ccg_parg_path = sys.argv[1]

    t0 = time.time()
    ccg_auto_ids = get_ccg_auto_ids(CCG_AUTO)
    print(f"get_ccg_auto_ids took: {time.time()-t0}s")

    t0 = time.time()
    ccg_parg = get_ccg_parg(ccg_parg_path, ccg_auto_ids)
    print(f"generate_files took: {time.time()-t0}s")

    t0 = time.time()
    generate_files(ccg_parg)
    print(f"generate_files took: {time.time()-t0}s")


def get_ccg_auto_ids(ccg_auto_path):
    ids = []
    with open(ccg_auto_path, "r") as read_file:
        data = read_file.readlines()
        for elem in data:
            if elem.startswith("ID="):
                id = elem.split("ID=")[1].split(" ")[0]
                ids.append(id)
    return ids


def generate_files(ccg_parg):
    generate_ccg_files(ccg_parg)


def generate_ccg_files(ccg_pargs):
    with open(CCG_PARG, "w") as f:
        for parg in ccg_pargs:
            f.write(parg)


def get_file_list(ccg_parg_path):
    file_list = []
    for root, _, files in os.walk(ccg_parg_path):
        for file in files:
            if file.startswith("."):
                continue
            file_list.append(os.path.join(root, file))
    file_list.sort()

    return file_list


def get_ccg_parg(ccg_parg_path, auto_ids):
    file_list = get_file_list(ccg_parg_path)
    ccg_parg = []

    for f in tqdm.tqdm(file_list[:10]):
        with open(f, "r") as read_file:
            data = read_file.readlines()
            current_id = None
            for elem in data:
                if elem.startswith("<s id="):
                    id = elem.split(" ")[1].split("id=")[1].split('"')[1]
                    if id != current_id:
                        current_id = id
                if current_id in auto_ids:
                    ccg_parg.append(elem)

    return ccg_parg


if __name__ == "__main__":
    main()
