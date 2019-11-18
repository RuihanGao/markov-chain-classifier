import re
import pickle
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
print("log_file_path", log_file_path)

output_file = log_file_path.replace('.log', '_acc.pkl')
# initialization
result = np.zeros((4, 400)) # refer to distill_main.py
regex = 'Testing novel+old'
with open(log_file_path, "r") as fin:
    match_list = []
    for line in fin:
        if regex in line:
            # print(line)
            # print(line.split('\t'))
            info = line.split('\t')
            period = int(info[0].split('Period:')[1])
            iteration = int(info[1].split(':')[1])
            acc = float(info[8].split('=')[1])
            # print("period: ", period, type(period)) 
            # print("iteration: ", iteration, type(iteration))
            # print ("acc: ", acc, type(acc))
            result[period][iteration] = acc
            # raise ValueError("stop here to check")
fin.close()

fout = open(output_file, "wb")
pickle.dump(result, fout)
