import logging
import os
import pickle
import time
from collections import defaultdict

import coloredlogs
import grounded_ccg_inducer.constants as constants
import tqdm
from grounded_ccg_inducer.dataloader import Dataloader
from grounded_ccg_inducer.embeddings import Embeddings
from grounded_ccg_inducer.pos_analyzer import POSAnalyzer
from video_recognizer.action_recognizer import ActionRecognizer
from video_recognizer.object_recognizer import ObjectRecognizer

DEBUG = False
dataset_name = None


def main():
    # Import DiDeMo Dataset
    logging.info("Loading DiDeMo...")
    dataloader = Dataloader(constants.didemo_path)

    logging.info("---NO ENRICHMENT---")
    object_recognizer, action_recognizer = get_models(
        use_enrichment=False, use_embeddings=False
    )
    logging.info(len(object_recognizer.imagenet_classes))

    pos_analyzer = POSAnalyzer()

    t0 = time.time()

    categorize_words(object_recognizer, action_recognizer, pos_analyzer, dataloader)

    t1 = time.time()
    logging.info(t1 - t0)

    logging.info("---HYPER/HYPONYMS---")

    object_recognizer, action_recognizer = get_models(
        use_enrichment=True, use_embeddings=False
    )
    logging.info(len(object_recognizer.imagenet_classes))

    pos_analyzer = POSAnalyzer()

    t0 = time.time()

    categorize_words(object_recognizer, action_recognizer, pos_analyzer, dataloader)

    t1 = time.time()
    logging.info(t1 - t0)
    logging.info("---EMBEDDINGS---")

    object_recognizer, action_recognizer = get_models(
        use_enrichment=True, use_embeddings=True, top_n=40
    )
    logging.info(len(object_recognizer.imagenet_classes))

    pos_analyzer = POSAnalyzer()

    t0 = time.time()

    categorize_words(object_recognizer, action_recognizer, pos_analyzer, dataloader)

    t1 = time.time()
    logging.info(t1 - t0)


def get_models(use_enrichment=True, use_embeddings=False, top_n=None):
    action_recognizer = ActionRecognizer(use_enrichment=use_enrichment)
    if use_embeddings:
        embeddings = Embeddings(constants.fasttext_path, topn=top_n)
        object_recognizer = ObjectRecognizer(
            use_enrichment=use_enrichment, embeddings_model=embeddings
        )
    else:
        object_recognizer = ObjectRecognizer(use_enrichment=use_enrichment)

    return object_recognizer, action_recognizer


def pos_tag_dataset(dataloader, pos_analyzer):
    _tagged_data_cache = os.path.join(".cache", "pos_tagged_dataset.pickle")
    if os.path.exists(_tagged_data_cache):
        with open(_tagged_data_cache, "rb") as fp:
            new_dict = pickle.load(fp)
    else:
        logging.info("Looping through sentences...")
        new_dict = {}
        for data_key, data_val in tqdm.tqdm(
            dataloader.dataset.items(), total=len(dataloader.dataset)
        ):
            new_dict[data_key] = []
            for sentence in data_val:
                pos_sentence = pos_analyzer.analyze_sentence(sentence)
                new_dict[data_key].append(pos_sentence)
        with open(_tagged_data_cache, "wb") as fp:
            pickle.dump(new_dict, fp)

    dataloader.set_tagged_dataset(new_dict)
    dataloader.preprocess_dataset()

    return dataloader.tagged_dataset


def categorize_words(object_recognizer, action_recognizer, pos_analyzer, dataloader):
    total_counter = defaultdict(int)

    logging.info("Looping through sentences...")
    for data_key, data_val in tqdm.tqdm(
        dataloader.dataset.items(), total=len(dataloader.dataset)
    ):
        top_p_objects = []
        object_labels = []
        top_p_objects = object_recognizer.get_top_p(data_key, 0.1)
        object_labels = [label for (label, _) in top_p_objects]

        top_p_actions = []
        action_labels = []
        top_p_actions = action_recognizer.get_top_p(data_key, 0.1)
        action_labels = [label for (labels, _) in top_p_actions for label in labels]

        for sentence in data_val:
            pos_sentence = pos_analyzer.analyze_sentence(sentence)

            for word, pos in pos_sentence:
                if pos == "NOUN":
                    if word in object_labels:
                        total_counter["object_tp"] += 1
                    else:
                        total_counter["object_fn"] += 1
                else:
                    if word in object_labels:
                        total_counter["object_fp"] += 1
                    else:
                        total_counter["object_tn"] += 1

                if pos == "VERB":
                    if word in action_labels:
                        total_counter["action_tp"] += 1
                    else:
                        total_counter["action_fn"] += 1
                else:
                    if word in action_labels:
                        total_counter["action_fp"] += 1
                    else:
                        total_counter["action_tn"] += 1

            total_counter["total"] += len(sentence)

    print_dataset_statistics(total_counter)


def print_dataset_statistics(total_counter):
    object_recall = total_counter["object_tp"] / (
        total_counter["object_tp"] + total_counter["object_fp"]
    )
    object_precision = total_counter["object_tp"] / (
        total_counter["object_tp"] + total_counter["object_fn"]
    )

    action_recall = total_counter["action_tp"] / (
        total_counter["action_tp"] + total_counter["action_fp"]
    )
    action_precision = total_counter["action_tp"] / (
        total_counter["action_tp"] + total_counter["action_fn"]
    )

    logging.info("Total words: %s", total_counter["total"])

    logging.info("Object precision: %.4f", object_recall)
    logging.info("Object recall: %.4f", object_precision)

    logging.info("Action precision: %.4f", action_recall)
    logging.info("Action recall: %.4f", action_precision)


def init(debug=False):
    level = "INFO"
    if debug:
        level = "DEBUG"
    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_format_str = "> %(asctime)s %(levelname)-8s %(message)s"
    log_formatter = logging.Formatter(log_format_str)
    coloredlogs.install(level=level, fmt=log_format_str)
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(
        "{0}/{1}.txt".format("logs", f"test_object_action_recognizer_log_{timestr}")
    )
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)


if __name__ == "__main__":
    init(debug=DEBUG)
    main()
