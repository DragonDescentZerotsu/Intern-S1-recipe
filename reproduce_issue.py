import os
import sys

if os.environ.get('PYTHONHASHSEED') != '42':
    os.environ['PYTHONHASHSEED'] = '42'
    os.execv(sys.executable, [sys.executable] + sys.argv)

from tdc.multi_pred.ppi import PPI
import numpy as np
import random

np.random.seed(42)
random.seed(42)

def get_data_sample():
    data = PPI(name='HuRI').neg_sample(frac=1)
    split = data.get_split()
    test_inputs = split['test'][['Protein1', 'Protein2']].values
    return test_inputs[0] # Return the first sample

print("\nRun 1:")
sample1 = get_data_sample()
print(sample1)

print("\nRun 2:")
sample2 = get_data_sample()
print(sample2)

if np.array_equal(sample1, sample2):  # .split('*')[0]
    print("Samples are identical.")
else:
    print("Samples are different.")
