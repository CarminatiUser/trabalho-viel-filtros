from shutil import rmtree
from os.path import exists
from os import mkdir

def create_output_dir(dir: str):
    if exists(dir):
        rmtree(dir)
    mkdir(dir)