import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

# init_opt = 'diff_init'

# parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-init", "--init_opt", required=True, help="init_opt specification for model path adn dataset")
args = parser.parse_args()

init_opt = args.init_opt
# check the argument
print("init_opt", init_opt)
log_file_path = "distill_main_{}.log".format(init_opt)

