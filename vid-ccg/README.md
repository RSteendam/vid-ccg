# Vid-CCG

Vid-CCG is a visually grounded unsupervised CCG inducer, written for the AI master thesis of Ruben Steendam. This Inducer relies on object/action information instead of POS tags. This project consists of two subprojects: inducer and recognizer.

Inducer is a partly replication of the CCG-Induction. We made this as a proof of concept, but do not use this in further experiments, since the code is slower than the original CCG-Induction.

Recognizer changes the way CCG-Induction works. Instead of relying on POS tags, we rely on object/action information. All experiments and setup are described in the [Recognizer README](vid-ccg/README.md).

# grounded-ccg-inducer
Visually grounded unsupervised CCG inducer in Python. Written for the Master Thesis AI of Ruben Steendam.

## Setup
Most of the setup is automated in the scripts below. You only need to download the tri-headfirst-model manually and give this as an argument to `inital_setup`.

```
# manually download tri model from https://drive.google.com/file/d/1mxl1HU99iEQcUYhWhvkowbE4WOH0UKxv/view?usp=sharing
# manually clone the CCG Inducer from Bisk in the parent directory of the grounded-ccg-inducer. I.e. if the path for the grounded-ccg-inducer is /home/user/github/grounded-ccg-inducer, then clone bisk in /home/user/github/CCG-Induction (https://github.com/ybisk/CCG-Induction)
# manually download and extract DiDeMo features train & val from https://www.robots.ox.ac.uk/~vgg/challenges/video-pentathlon/challenge.html. Place raw-captions.pkl in vid-ccg/data
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-2.txt
python3 -m scripts.initial_setup ~/location/to/tri_headfirst.tar.gz
# Install Maven
# Make dirs

```

## Experiments
### Generating dataset
To generate a dataset we use `make_datasets`:
```
python3 -m scripts.make_datasets settings/make_datasets/experiment_settings_40_noisy.yaml
```

To generate all datasets from the settings folder we use `make_datasets_runner`
```
python3 -m scripts.make_datasets_runner settings/make_datasets/
```

### Preprocess and analyze data
To analyze and preprocess data we run `preprocess_and_analyze_dataset`
```
python3 -m scripts.preprocess_and_analyze_dataset
```

### Generate Silver trees
To generate silver trees we run `generate_silver_trees` on output/dataset_sentences from make_datasets. 
On newer Linux systems you might encounter a `python not found` error. In that case make sure that python is an alias for python3, i.e. by installing `python-is-python3`. Or edit the depccg bin to use `python3`.
```
python3 -m scripts.generate_silver_trees output/dataset_sentences
```

### Run induction
To induce a grammar we run `induce_grammar`.
```
python3 -m scripts.induce_grammar settings/induction/induction_config_40_noisy_object_action_closed_other_open.properties
```

To induce all grammars from the settings folder we use `induce_grammar_runner`.
```
python3 -m induce_grammar
```

If for some reason you need to run a different test on a trained model, you can look at `settings/induction/test.properties` to see how to do this.


## Testing pipeline
Testing is a little harder, since there are no gold trees. The gold trees are generated with depccg (https://github.com/masashi-y/depccg). For accuracy comparison we use SuperTagAccuracy from Bisk.

### SuperTagAccuracy
Generate a pos file like [example input file](examples/induction_input_example.txt) -> comes out of make_datasets

Run the Bisk inducer with the [example config](examples/induction_config_example.properties)

Retrieve the [output file](examples/induction_output_example.JSON.gz)

Run 
```
java -cp target/CCGInduction-1.0-jar-with-dependencies.jar \
CCGInduction.evaluation.SupertagAccuracy \
gold=depccg_auto_output_example.txt \
system=induciton_output_example.JSON.gz
```

## CCGBank testing
merge_ccgbank_parg relies on CCG_AUTO = "ccg_auto_gold.auto"
CCG_AUTO can be generated with merge_ccgbank_auto
1. Generate PARG files for dependency testing

# Vid-CCG - Inducer

The Inducer from Vid-CCG is a first step in replicating the Inducer from Bisk et al. The induction step is fully operational, but the other steps are missing. Some development work for Inside-Outside (or CYK) has been done. But the code was deemed to slow, to fully finish the programming. 

Even the induction step is quite a lot slower when compared to the Java implementation. Therefore this module should only be used as a starting point for 1. understanding how the inducer works and 2. building a ccg-inducer in Python.

Possible upgrades could be:
- Increasing speed
- Incorporating a model (HDP)
- Better logging and saving
- Fully supporting IO/CYK


Command
`
python3 -m run_inducer output/dataset_baseline_small.txt && python3 -m scripts.induce_grammar settings/induction_config_didemo.properties && gzip -dkf output/induction/didemo/Lexicon.gz && python3 -m test_lexicon output/python_lexicon output/induction/didemo/Lexicon
`
## CCGBank testing
merge_ccgbank_parg relies on CCG_AUTO = "ccg_auto_gold.auto"
CCG_AUTO can be generated with merge_ccgbank_auto
1. Generate PARG files for dependency testing

1. Generate the PTB and CCGBank files needed: `python3 -m wsj.merge_ccgbank_auto data/CCGrebank_v1.0/data/AUTO/`
2. Depccg evaluation with: `python3 -m scripts.generate_silver_trees output/wsj/wsj_sentences.txt` and `python3 -m scripts.evaluate_supertag output/wsj/ccg_auto_gold.auto output/depccg/auto_wsj_sentences.txt`
3. SpaCy evaluation with: `python3 -m wsj.pos_tag_wsj`
3. Merge CCGBank PARG`