import logging
import os
import sys

from . import script_utils

DEBUG = False


def main():
    system = sys.argv[1]
    gold = sys.argv[2]
    logging.info(f"running system: {system} and gold: {gold}")
    wd = os.getcwd()
    gold_path = os.path.join(wd, gold)
    system_path = os.path.join(wd, system)
    java_dir = "../CCG-Induction/"

    undirected_eval_command = f"java -cp target/CCGInduction-1.0-jar-with-dependencies.jar \
        CCGInduction.evaluation.CoNLLDependencies \
        gold={gold_path} \
        system={system_path} \
        -m=Undirected"

    directed_eval_command = f"java -cp target/CCGInduction-1.0-jar-with-dependencies.jar \
        CCGInduction.evaluation.CoNLLDependencies \
        gold={gold_path} \
        system={system_path} \
        -m=Directed"

    script_utils.maven_build()
    script_utils.run_command(undirected_eval_command, command_dir=java_dir)
    script_utils.run_command(directed_eval_command, command_dir=java_dir)
    print()


if __name__ == "__main__":
    script_utils.init_log("evaluate_conll_log", debug=DEBUG)
    main()
