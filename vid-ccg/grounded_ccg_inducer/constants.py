import os

_data_dir = "data"
_secrets_dir = "secrets"

_didemo_large_filename = "didemo-raw-captions-release-1.pkl"
_didemo_small_filename = "didemo-raw-captions-release-2.pkl"
_didemo_filename = _didemo_large_filename
_imagenet_filename = "imagenet_classes.txt"
_kinetics_filename = "kinetics_700_labels.txt"
_fasttext_filename = "wiki-news-300d-1M.vec"
_feature_perms_filename = "feature_perms.pkl"

didemo_path = os.path.join(_data_dir, _didemo_filename)
imagenet_path = os.path.join(_data_dir, _imagenet_filename)
kinetics_path = os.path.join(_data_dir, _kinetics_filename)
fasttext_path = os.path.join(_data_dir, _fasttext_filename)
feature_perms_path = os.path.join(_data_dir, _secrets_dir, _feature_perms_filename)

object_model_key = "imagenet.resnext101_32x48d"
object_model_path = os.path.join(_data_dir, "resnext101_32x48d-max-logits.pickle")
action_model_key = "i3d.i3d"
action_model_path = os.path.join(_data_dir, "i3d-max-logits.pickle")


use_cache = True
use_enriched = False
top_n = 40
