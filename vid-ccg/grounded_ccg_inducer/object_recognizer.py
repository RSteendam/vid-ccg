import itertools
import logging
import os
import pickle

import grounded_ccg_inducer.constants as constants
import nltk
import tqdm
from nltk.corpus import wordnet as wn


class ObjectRecognizer:
    def __init__(self, use_enrichment=False, embeddings_model=None) -> None:
        logging.info("Recognizing objects with %s...", constants._imagenet_filename)
        self.imagenet_filename = constants.imagenet_path
        self.use_enrichment = use_enrichment
        self.imagenet_classes = self.load_imagenet_classes()

        logging.info("Nr of imagenet classes: %s", len(self.imagenet_classes))
        self.enriched_imagenet = self.imagenet_to_dict()
        if self.use_enrichment:
            self.enriched_imagenet = self.enrich_imagenet_hyper_and_hyponyms()
            self.embeddings_model = embeddings_model
            if self.embeddings_model is not None:
                print("embeddings model found")
                self.enriched_imagenet = self.enrich_imagenet_embeddings()

    def load_imagenet_classes(self):
        with open(self.imagenet_filename) as f:
            imagenet_classes = [s.strip() for s in f.readlines()]

        return imagenet_classes

    def enrich_imagenet_hyper_and_hyponyms(self):
        enriched_imagenet = dict()
        for word in self.imagenet_classes:
            superset = get_hyper_and_hyponyms(word)
            enriched_imagenet[word] = superset

        return enriched_imagenet

    def imagenet_to_dict(self):
        enriched_imagenet = dict()
        for word in self.imagenet_classes:
            enriched_imagenet[word] = set([word])
        for word in self.imagenet_classes:
            if word not in enriched_imagenet:
                print(f"HELLO {word} is not in enriched set")
        logging.info("Nr of deduplicated imagenet classes: %s", len(enriched_imagenet))
        return enriched_imagenet

    def enrich_imagenet_embeddings(self):
        _imagenet_embedded_cache = os.path.join(
            ".cache", f"imagenet_embedded_top_n_{self.embeddings_model.topn}.pickle"
        )
        if self.embeddings_model is None or self.use_enrichment is False:
            return self.enriched_imagenet
        else:
            if os.path.exists(_imagenet_embedded_cache) and constants.use_cache:
                logging.info(f"Using cache for embeddings: {_imagenet_embedded_cache}")
                with open(_imagenet_embedded_cache, "rb") as fp:
                    enriched_dict = pickle.load(fp)
            else:
                logging.info(
                    f"Not using cache for embeddings: {_imagenet_embedded_cache}"
                )
                found = 0
                not_found = 0
                enriched_dict = {}
                logging.debug("imagenet classes %s", self.imagenet_classes)
                for label in tqdm.tqdm(
                    self.imagenet_classes, total=len(self.imagenet_classes)
                ):
                    new_enriched_labels = set([label])
                    for enriched_label in [label]:
                        try:
                            most_similar = self.embeddings_model.most_similar(
                                enriched_label
                            )
                            found += 1
                            most_similar = [x for (x, _) in most_similar]
                            for word in most_similar:
                                _, tag = nltk.tag.pos_tag([word], tagset="universal")[0]
                                if tag == "NOUN":
                                    logging.debug("new label is NOUN")
                                    new_enriched_labels.update([word])
                        except Exception as e:
                            logging.info(e)
                            logging.info(
                                "could not find '%s' in embeddings", enriched_label
                            )
                            not_found += 1

                    enriched_dict[label] = new_enriched_labels
                with open(_imagenet_embedded_cache, "wb") as fp:
                    pickle.dump(enriched_dict, fp)
                logging.info(f"Did not find embeddings for {not_found} words.")
            return enriched_dict


def get_hyper_and_hyponyms(word):
    word = word.replace("'", "")
    word = word.replace(" ", "_")
    superset = [word]
    synsets = wn.synsets(word)
    if len(synsets) == 0:
        return superset
    synset = synsets[0]
    hyponyms = synset.hyponyms()
    hypernyms = synset.hypernyms()

    hyponyms_lemma = [hyponym.lemma_names() for hyponym in hyponyms]
    hyponyms_lemma = list(itertools.chain.from_iterable(hyponyms_lemma))
    hypernyms_lemma = [hypernym.lemma_names() for hypernym in hypernyms]
    hypernyms_lemma = list(itertools.chain.from_iterable(hypernyms_lemma))
    superset.extend(hyponyms_lemma)
    superset.extend(hypernyms_lemma)

    return superset
