import itertools
import json
import logging
import os
import pickle

import grounded_ccg_inducer.constants as constants
import nltk
import torch
import tqdm
from nltk.corpus import wordnet as wn
from video_recognizer.utils import load_inverse_permutation


class ObjectRecognizer():
    def __init__(self, use_enrichment=False, embeddings_model=None) -> None:
        logging.info("Recognizing objects with %s...", constants.object_model_key)
        logging.debug("%s %s", constants.object_model_key, constants.object_model_path)
        self.imagenet_filename = constants.imagenet_path
        self.use_enrichment = use_enrichment
        self.imagenet_classes = self.load_imagenet_classes()
        first5 = self.imagenet_classes[:10]
        logging.debug("IMAGENET CLASSES")
        logging.debug(first5)
        self.enriched_imagenet = self.imagenet_classes
        if self.use_enrichment:
            self.enriched_imagenet = self.enrich_imagenet_hyper_and_hyponyms()
            logging.debug("HYPER/HYPONYMS")
            for label in first5:
                logging.debug(f"{label}, {self.enriched_imagenet[label]}")
        if self.use_enrichment:
            self.embeddings_model = embeddings_model
            if self.embeddings_model is not None:
                self.enriched_imagenet_embeddings = self.enrich_imagenet_embeddings()
                logging.debug("EMBEDDINGS")
                for label in first5:
                    logging.debug(f"{label}, {self.enriched_imagenet_embeddings[label]}")
        self.model_key = constants.object_model_key
        self.model = self.load_model(constants.object_model_path)
        self.inverse_permutation = load_inverse_permutation(constants.feature_perms_path, self.model_key)


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
    
    def enrich_imagenet_embeddings(self):
        logging.info("len imagenet: %s", len(self.enriched_imagenet))
        _imagenet_embedded_cache = os.path.join(".cache", f'experimental_imagenet_embedded_top_n_{self.embeddings_model.topn}.pickle')
        if self.embeddings_model is None or self.use_enrichment is False:
            logging.info("object recognizer: using hyper and hyponyms")
            return self.enriched_imagenet
        else:
            if os.path.exists(_imagenet_embedded_cache):
                logging.info("Using cache for embeddings: %s", _imagenet_embedded_cache)
                with open(_imagenet_embedded_cache, 'rb') as fp:
                    enriched_dict = pickle.load(fp)
            else:
                logging.info("Not using cache for embeddings: %s", _imagenet_embedded_cache)
                found = 0
                not_found = 0
                enriched_dict = {}
                logging.debug("imagenet classes %s", self.imagenet_classes)
                for label in tqdm.tqdm(self.imagenet_classes, total=len(self.imagenet_classes)):

                    new_enriched_labels = set([label])
                    for enriched_label in [label]:
                        try:
                            most_similar = self.embeddings_model.most_similar(enriched_label)
                            found += 1
                            most_similar = [x for (x, _) in most_similar]
                            for word in most_similar:
                                _, tag = nltk.tag.pos_tag([word], tagset="universal")[0]
                                if tag == "NOUN":
                                    logging.debug("new label is NOUN")
                                    new_enriched_labels.update([word])
                        except Exception as e:
                            logging.debug(e)
                            logging.debug("could not find '%s' in embeddings", enriched_label)
                            not_found += 1
                    enriched_dict[label] = new_enriched_labels
                with open(_imagenet_embedded_cache, 'wb') as fp:
                    pickle.dump(enriched_dict, fp)
            logging.info("len imagenet: %s", len(self.enriched_imagenet))
            logging.info("len imagenet: %s", len(enriched_dict))
            return enriched_dict
                    


    def _get_imagenet_classes(self, didemo_key):
        return self.model[didemo_key][:, self.inverse_permutation]
    
    def load_model(self, model_filename):
        with open(model_filename, "rb") as f:
            model = pickle.load(f)
        logging.debug(len(model))
        return model

    def get_top_p(self, didemo_key, threshold):
        features = self._get_imagenet_classes(didemo_key)
        output = torch.from_numpy(features).float()
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Show top categories per image
        topk_prob, topk_catid = torch.topk(probabilities, 100)

        top_p = []
        

        for i in range(topk_prob.size(0)):
            if topk_prob[i].item() >= threshold:

                if self.use_enrichment:
                    if self.embeddings_model is not None:
                        for word in self.enriched_imagenet_embeddings[self.imagenet_classes[topk_catid[i]]]:
                            top_p.append((word, topk_prob[i].item()))
                    else:
                        logging.debug(self.enriched_imagenet[self.imagenet_classes[topk_catid[i]]])
                        for word in self.enriched_imagenet[self.imagenet_classes[topk_catid[i]]]:
                            top_p.append((word, topk_prob[i].item()))

                else:
                    top_p.append((self.imagenet_classes[topk_catid[i]], topk_prob[i].item()))
            else:
                break
        
        return top_p


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

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)