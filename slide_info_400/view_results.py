#!/usr/bin/python

import sys
import pickle 

print("Number of arguments: ", len(sys.argv))

print("load pkl file ", str(sys.argv[1]))
results = pickle.load(open(str(sys.argv[1]), 'rb'))
print(results)
