import logging
import sys
import time
from collections import defaultdict

import grounded_ccg_inducer.constants as constants
import tqdm
import yaml
from grounded_ccg_inducer.action_recognizer import ActionRecognizer
from grounded_ccg_inducer.dataloader import Dataloader
from grounded_ccg_inducer.embeddings import Embeddings
from grounded_ccg_inducer.object_recognizer import ObjectRecognizer
from grounded_ccg_inducer.pos_analyzer import POSAnalyzer
from scripts import script_utils

DEBUG = False
dataset_name = None
settings_list = {}


def main():
    settings_yaml = sys.argv[1]
    with open(settings_yaml) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        global settings_list
        settings_list = yaml.load(file, Loader=yaml.FullLoader)

    logging.info(f"running experiment: {settings_list['experiment_name']}")
    logging.info(f"with settings: {settings_list}")

    # Import DiDeMo Dataset
    logging.info("Loading DiDeMo...")
    dataloader = Dataloader(constants.didemo_path)

    object_recognizer, action_recognizer = get_models(
        use_enrichment=True, use_embeddings=True, top_n=40
    )
    flattened_objects = set()
    for key, val in object_recognizer.enriched_imagenet.items():
        flattened_objects.update(val)
    logging.info(f"Nr of enriched Imagenet classes: {len(flattened_objects)}")
    flattened_actions = set()
    for _, val in action_recognizer.enriched_kinetics.items():
        flattened_actions.update(val)
    logging.info(f"Nr of enriched Imagenet classes: {len(flattened_actions)}")

    pos_analyzer = POSAnalyzer()

    t0 = time.time()
    new_dataset = script_utils.pos_tag_dataset(dataloader, pos_analyzer)

    dataset_dict = make_datasets(
        new_dataset=new_dataset,
        flattened_object_set=flattened_objects,
        flattened_action_set=flattened_actions,
    )
    filter_type = "perfect" if settings_list["filter_pos"] else "noisy"
    tagged_data_output = f"{settings_list['top_n']}_embeddings_{filter_type}"
    for name, dataset in dataset_dict.items():
        new_dataset = []
        for sent in dataset:
            sent = [(word, tag) for (word, tag) in sent if tag != "SPACE"]
            new_dataset.append(sent)

        output_name = "output/dataset"
        if not name.startswith("baseline"):
            output_name += f"_{tagged_data_output}"

        output_name += f"_{name}.txt"
        with open(output_name, "w") as f:
            for sent in new_dataset:
                for i, (word, tag) in enumerate(sent):
                    f.write(f"{i+1}\t{word}\t_\t_\t_\t{tag}\n")
                f.write("\n")

    with open("output/dataset_sentences.txt", "w") as f:
        longest_sent = []
        for sent in new_dataset:
            sent_len = len(sent)
            if sent_len > len(longest_sent):
                longest_sent = sent
            sentence_string = " ".join([word for (word, _) in sent])
            f.write(f"{sentence_string}\n")
        print(f"Longest sent has length: {len(longest_sent)} with sent {longest_sent}")

    t1 = time.time()
    logging.info(f"Time to run program: {t1-t0}")


def get_models(use_enrichment=True, use_embeddings=False, top_n=None):
    action_recognizer = ActionRecognizer(use_enrichment=use_enrichment)
    if use_embeddings:
        print("hello")
        embeddings = Embeddings(constants.fasttext_path, topn=top_n)
        object_recognizer = ObjectRecognizer(
            use_enrichment=use_enrichment, embeddings_model=embeddings
        )
    else:
        print("gbye")
        object_recognizer = ObjectRecognizer(use_enrichment=use_enrichment)

    return object_recognizer, action_recognizer


def make_datasets(new_dataset, flattened_object_set, flattened_action_set):
    dataset_names = [
        "baseline",
        "baseline_noun_verb",
        "baseline_noun_verb_closed",
        "object_action",
        "object_action_closed",
        "object_action_closed_open",
    ]
    dataset_dict = {name: [] for name in dataset_names}

    open_class_tags = {"ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"}
    closed_class_tags = {
        "ADP",
        "AUX",
        "CCONJ",
        "DET",
        "NUM",
        "PART",
        "PRT",
        "PRON",
        "SCONJ",
        "CONJ",
    }
    object_action_tags = {"object", "action"}
    noun_verb_tags = {"NOUN", "VERB"}
    tagsets = {
        "object-action": object_action_tags,
        "object-action-closed": object_action_tags.union(closed_class_tags),
        "object-action-closed-open": object_action_tags.union(
            closed_class_tags, open_class_tags
        ),
        "noun-verb": noun_verb_tags,
        "noun-verb-closed": noun_verb_tags.union(closed_class_tags),
    }
    print(tagsets["noun-verb"])

    total_counter = defaultdict(int)
    
    logging.info("Looping through sentences...")
    detected_objects = 0
    detected_actions = 0
    for sentence in tqdm.tqdm(new_dataset, total=len(new_dataset)):
        tagged_sentence = {name: [] for name in dataset_names}
        # Loop through senteneces and get words and pos tags
        for word, pos in sentence:
            if pos == "NOUN":
                total_counter["noun"] += 1
                if word in flattened_object_set:
                    total_counter["object_tp"] += 1
                else:
                    total_counter["object_fn"] += 1
            else:
                if word in flattened_object_set and not settings_list["filter_pos"]:
                    total_counter["object_fp"] += 1
                else:
                    total_counter["object_tn"] += 1

            if pos == "VERB":
                total_counter["verb"] += 1
                if word in flattened_action_set:
                    if word not in flattened_object_set:
                        total_counter["action_tp"] += 1
                else:
                    total_counter["action_fn"] += 1
            else:
                if word in flattened_action_set and not settings_list["filter_pos"]:
                    if word not in flattened_object_set:
                        total_counter["action_fp"] += 1
                else:
                    total_counter["action_tn"] += 1
            tag = pos
            new_tag = pos

            # If tag is NOUN or ACTION, set new_tag to UNK and increase counter
            if pos == "NOUN":
                new_tag = "UNK"

            elif pos == "VERB":
                new_tag = "UNK"

            # If word is detected
            if word in flattened_action_set:
                detected_actions += 1
                if settings_list["filter_pos"] and pos != "VERB":
                    pass
                # else update new_tag (so new tag gets updated if we do perfect analysis and are correct or if we don't do perfect analysis)
                else:
                    new_tag = "action"

            # If a word is both in the object and action set, overwrite tag with object.
            if word in flattened_object_set:
                if new_tag == "action":
                    print(word)
                    detected_actions -= 1
                detected_objects += 1
                if settings_list["filter_pos"] and pos != "NOUN":
                    pass
                else:
                    new_tag = "object"

            new_tag_object_action = new_tag
            new_tag_object_action_closed = new_tag
            new_tag_object_action_closed_open = new_tag
            new_tag_noun_verb = tag
            new_tag_noun_verb_closed = tag

            if new_tag not in tagsets["object-action"]:
                new_tag_object_action = "UNK"
            if new_tag not in tagsets["object-action-closed"]:
                new_tag_object_action_closed = "UNK"
            if new_tag not in tagsets["object-action-closed-open"]:
                new_tag_object_action_closed_open = "UNK"
            if tag not in tagsets["noun-verb"]:
                new_tag_noun_verb = "UNK"
            if tag not in tagsets["noun-verb-closed"]:
                new_tag_noun_verb_closed = "UNK"

            tagged_sentence["baseline"].append((word, tag))
            tagged_sentence["object_action"].append((word, new_tag_object_action))
            tagged_sentence["object_action_closed"].append(
                (word, new_tag_object_action_closed)
            )
            tagged_sentence["object_action_closed_open"].append(
                (word, new_tag_object_action_closed_open)
            )
            tagged_sentence["baseline_noun_verb"].append((word, new_tag_noun_verb))
            tagged_sentence["baseline_noun_verb_closed"].append(
                (word, new_tag_noun_verb_closed)
            )

        total_counter["total"] += len(sentence)

        for name in dataset_names:
            dataset_dict[name].append(tagged_sentence[name])

    for sentence in dataset_dict["object_action"]:
        for _, tag in sentence:
            total_counter[tag] += 1

    print(f"objects: {detected_objects}")
    print(f"actions: {detected_actions}")
    print_dataset_statistics(total_counter)

    return dataset_dict


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
    
    logging.info("Total nouns: %s", total_counter["noun"])
    logging.info("Total verbs: %s", total_counter["verb"])

    logging.info("Object precision: %.4f", object_recall)
    logging.info("Object recall: %.4f", object_precision)

    logging.info("Action precision: %.4f", action_recall)
    logging.info("Action recall: %.4f", action_precision)


if __name__ == "__main__":
    script_utils.init_log("make_datasets", debug=DEBUG)
    main()
