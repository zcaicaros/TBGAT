import numpy as np
from numpy import genfromtxt
import torch
from tqdm import tqdm

pbar = tqdm([i for i in range(1)])
for char in pbar:
    pbar.set_description("Processing %s" % char)



pbar = tqdm([i for i in range(100000)])
for ichar in pbar:
    pbar.set_postfix({'num_vowels': ichar})