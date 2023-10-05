import os
import sys

from scripts import script_utils

DEBUG = False


def main():
    gold = sys.argv[1]
    system = sys.argv[2]
    wd = os.getcwd()
    gold_path = os.path.join(wd, gold)
    system_path = os.path.join(wd, system)
    java_dir = "../CCG-Induction/"

    java_command = f"java -cp \
        target/CCGInduction-1.0-jar-with-dependencies.jar \
        CCGInduction.evaluation.SupertagAccuracy \
        gold={gold_path} \
        system={system_path} \
        -removePunct"

    script_utils.maven_build()
    script_utils.run_command(java_command, command_dir=java_dir)


if __name__ == "__main__":
    script_utils.init_log("evaluate_supertag", debug=DEBUG)
    main()
