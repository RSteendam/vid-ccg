"""
Utils for scripts.
"""
import logging
import os
import pickle
import subprocess
import sys
import time
from typing import IO, Union

import coloredlogs
import tqdm
from grounded_ccg_inducer.dataloader import Dataloader
from grounded_ccg_inducer.pos_analyzer import POSAnalyzer


def maven_build() -> None:
    """
    Build CCG-Induction with Maven.
    """
    command_dir = "../CCG-Induction"
    mvn_command = "mvn package -DskipTests"
    returncode, _ = run_command(mvn_command, command_dir=command_dir, debug=True)

    if returncode == 0:
        logging.info("Build succesful")
    else:
        raise RuntimeError("Maven build failed")


def run_command(
    command: str, command_dir: str = "", debug: bool = False
) -> tuple[int, list[str]]:
    """
    Helper to run subprocess commands.
    """
    working_dir = os.getcwd()

    if command_dir != "":
        os.chdir(command_dir)

    with subprocess.Popen(
        command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ) as process:
        output = []

        def _log_subprocess_output(
            pipe: Union[IO[bytes], None], debug: bool = False
        ) -> list[str]:
            """
            Log subprocess output to logging and return decoded output as list.
            """
            output: list[str] = []
            for line in pipe:  # b'\n'-separated lines
                decoded = line.decode("utf-8").rstrip()

                if debug:
                    logging.debug(decoded)
                else:
                    logging.info(decoded)
                output.append(decoded)
            return output

        for c in iter(lambda: process.stdout.read(1), b""):
            sys.stdout.buffer.write(c)
            sys.stdout.flush()
            output.append(c)

    os.chdir(working_dir)

    return (process.returncode, output)


def make_dir(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def init_log(task_name: str, debug: bool = False) -> None:
    level = "INFO"
    if debug:
        level = "DEBUG"
    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_format_str = "> %(asctime)s %(levelname)-8s %(message)s"
    log_formatter = logging.Formatter(log_format_str)
    coloredlogs.install(level=level, fmt=log_format_str)
    root_logger = logging.getLogger()
    dirs = ["logs", ".cache", "output", "output/depccg"]
    for dir in dirs:
        make_dir(dir)
    file_handler = logging.FileHandler(
        "{0}/{1}.txt".format("logs", f"{task_name}_log_{timestr}")
    )
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)


def pos_tag_dataset(
    dataloader: Dataloader, pos_analyzer: POSAnalyzer
) -> list[list[tuple[str, str]]]:
    """
    Helper for tagging the dataset.
    """
    _tagged_data_cache = os.path.join(".cache", "pos_tagged_dataset.pickle")
    if os.path.exists(_tagged_data_cache):
        with open(_tagged_data_cache, "rb") as fp:
            new_dict: dict[str, list[list[tuple[str, str]]]] = pickle.load(fp)
    else:
        logging.info("Looping through sentences...")
        new_dict = {}
        for data_key, data_val in tqdm.tqdm(
            dataloader.dataset.items(), total=len(dataloader.dataset)
        ):
            new_dict[data_key] = []
            for sentence in data_val:
                pos_sentence = pos_analyzer.analyze_sentence(sentence)
                new_dict[data_key].append(pos_sentence)
        with open(_tagged_data_cache, "wb") as fp:
            pickle.dump(new_dict, fp)

    dataloader.set_tagged_dataset(new_dict)
    dataloader.preprocess_dataset(debug=False)

    return dataloader.tagged_dataset
