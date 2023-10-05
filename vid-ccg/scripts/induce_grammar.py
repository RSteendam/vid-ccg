import os
import sys

from . import script_utils

DEBUG = False


def main():
    induction_config = sys.argv[1]
    wd = os.getcwd()
    induction_config_path = os.path.join(wd, induction_config)
    java_command = f"java -Xmx12g -jar target/CCGInduction-1.0-jar-with-dependencies.jar {induction_config_path}"

    java_dir = "../CCG-Induction/"

    script_utils.maven_build()
    script_utils.run_command(java_command, command_dir=java_dir)


if __name__ == "__main__":
    script_utils.init_log("induce_grammar", debug=DEBUG)
    main()
