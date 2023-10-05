import sys

from scripts.wsj import merge_wsj_ccgbank, remove_long_sentences, train_test_split


def main(ccg_auto_path):
    merge_wsj_ccgbank.main(ccg_auto_path=ccg_auto_path)
    train_test_split.main()


if __name__ == "__main__":
    ccg_auto_path = sys.argv[1]
    main(ccg_auto_path)
    exit()
