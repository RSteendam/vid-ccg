import logging
import pickle
import re

import grounded_ccg_inducer.constants as constants
import pandas as pd
import torch
from nltk.stem import WordNetLemmatizer
from pyinflect import getAllInflections
from video_recognizer.utils import load_inverse_permutation


class ActionRecognizer():
    def __init__(self,  use_enrichment=False) -> None:
        logging.info("Recognizing actions with %s...", constants.action_model_key)
        logging.debug("%s %s", constants.action_model_key, constants.action_model_path)
        self.use_enriched = use_enrichment
        self.kinetics_filename = constants.kinetics_path
        self.kinetics_classes = self.get_kinetics_classes()
        logging.debug(self.kinetics_classes)
        self.enriched_kinetics = self.parse_kinetics_classes()
        if self.use_enriched:
            self.enriched_kinetics = self.enrich_kinetics()
            logging.debug(self.enriched_kinetics)
        self.model_key = constants.action_model_key
        self.model = self.load_model(constants.action_model_path)
        self.inverse_permutation = load_inverse_permutation(constants.feature_perms_path, constants.action_model_key)

    def get_kinetics_classes(self):
        return pd.read_csv(self.kinetics_filename)['name'].values.tolist()
    
    def parse_kinetics_classes(self):
        kinetics_classes = {}
        
        for action in self.kinetics_classes:
            actions = set([action])
            res = re.sub('\([^\)]+\)', '', action)
            words = res.split(' ')
            for w in words:
                if w.endswith('ing'):
                    actions.add(w)
            kinetics_classes[action] = actions

        return kinetics_classes


    def enrich_kinetics(self):
        enriched_kinetics = {}
        lemmatizer = WordNetLemmatizer()
        
        for action in self.kinetics_classes:
            inflected_verbs = set([action])
            res = re.sub('\([^\)]+\)', '', action)
            words = res.split(' ')
            for w in words:
                if w.endswith('ing'):
                    inf = lemmatizer.lemmatize(w, 'v')
                    all_inflections = getAllInflections(inf, pos_type='V')
                    for key, val in all_inflections.items():
                        inflected_verbs.add(val[0])
            enriched_kinetics[action] = inflected_verbs

        return enriched_kinetics

    def load_model(self, model_filename):
        with open(model_filename, "rb") as f:
            model = pickle.load(f)
        logging.debug(len(model))
        return model

    def get_top_p(self, didemo_key, threshold):
        features = self._get_classes(didemo_key)
        output = torch.from_numpy(features).float()
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Show top categories per image
        topk_prob, topk_catid = torch.topk(probabilities, 100)

        top_p = []

        for i in range(topk_prob.size(0)):
            if topk_prob[i].item() >= threshold:
                top_p.append((self.enriched_kinetics[self.kinetics_classes[topk_catid[i]]], topk_prob[i].item()))
            else:
                break

        return top_p

    def _get_classes(self, didemo_key):
        return self.model[didemo_key][:, self.inverse_permutation]
