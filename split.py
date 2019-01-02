import glob
import os
import random

files = glob.glob('/home/ubuntu/book-corpus-doc/train/*/*')
files = sorted(files)
random.seed(0)
random.shuffle(files)
selected = files[:200]
for f in selected:
    os.rename(f, f.replace('train', 'dev'))
