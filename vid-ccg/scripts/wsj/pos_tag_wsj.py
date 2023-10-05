import logging
import time

import scripts.wsj.check_tagged_file as check_tagged_file
import spacy
import tqdm
from nltk.tag.mapping import map_tag
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex

PTB_SENTENCES = "output/wsj/wsj_train_sentences.txt"
PTB_SPACY = "output/wsj/wsj_dataset_spacy.txt"
PTB_GOLD = "output/wsj/wsj_out_train.txt"
DEBUG = False
dataset_name = None
settings_list = {}


def main():
    logging.info("running experiment: pos_tag_wsj")

    t0 = time.time()
    with open(PTB_SENTENCES) as f:
        ptb = [line.strip() for line in f.readlines()]
    new_dataset = pos_tag_dataset(ptb)

    with open(PTB_SPACY, "w") as f:
        for sent in new_dataset:
            for i, (word, tag) in enumerate(sent):
                f.write(f"{i+1}\t{word}\t_\t_\t_\t{tag}\n")
            f.write("\n")
    check_tagged_file.main(PTB_SPACY, PTB_GOLD)

    t1 = time.time()
    logging.info(f"Time to run program: {t1-t0}")


def pos_tag_dataset(dataset):
    logging.info("Looping through sentences...")
    new_dataset = []
    nlp = spacy.load("en_core_web_sm")

    def custom_tokenizer(nlp):
        inf = list(nlp.Defaults.infixes)  # Default infixes
        inf.remove(
            r"(?<=[0-9])[+\-\*^](?=[0-9-])"
        )  # Remove the generic op between numbers or between a number and a -
        inf = tuple(inf)  # Convert inf to tuple
        infixes = inf + tuple(
            [r"(?<=[0-9])[+*^](?=[0-9-])", r"(?<=[0-9])-(?=-)"]
        )  # Add the removed rule after subtracting (?<=[0-9])-(?=[0-9]) pattern
        infixes = [
            x for x in infixes if "-|–|—|--|---|——|~" not in x
        ]  # Remove - between letters rule
        infix_re = compile_infix_regex(infixes)

        return Tokenizer(
            nlp.vocab,
            prefix_search=nlp.tokenizer.prefix_search,
            suffix_search=nlp.tokenizer.suffix_search,
            infix_finditer=infix_re.finditer,
            token_match=nlp.tokenizer.token_match,
            rules=nlp.Defaults.tokenizer_exceptions,
        )

    nlp.tokenizer = Tokenizer(nlp.vocab)
    for sentence in tqdm.tqdm(dataset):
        doc = nlp(sentence)
        pos_sentence = [
            (token.text, map_tag("en-ptb", "universal", token.tag_)) for token in doc
        ]
        new_dataset.append(pos_sentence)

    return new_dataset


if __name__ == "__main__":
    main()
