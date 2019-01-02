import os
import subprocess
import time
import argparse
from subprocess import Popen

parser = argparse.ArgumentParser(description='Sentence tokenizer')
parser.add_argument('--num_parts', type=int, default=12,
                    help='Number of workers')
parser.add_argument('--dup_factor', type=int, default=5)
args = parser.parse_args()

N = args.num_parts
D = args.dup_factor
workers = []
for d in range(D):
    for i in range(N):
        process = Popen(['python', '/home/ubuntu/bert/create_pretraining_data.py',
                         '--vocab_file=%s/vocab.txt'%(os.environ['BERT_BASE_DIR']),
                         '--do_lower_case=True',
                         '--max_seq_length=512',
                         '--max_predictions_per_seq=78',
                         '--dupe_factor=1',
                         '--random_seed=%d'%d,
                         '--input_file=/home/ubuntu/book-corpus-doc/train/*/*',
                         '--output_file=/home/ubuntu/book-corpus-tf/train/%d.tfrecord.%d'%(i,d),
                         '--num_parts=%d'%N,
                         '--part_idx=%d'%i])
        workers.append(process)

for w in workers:
    w.wait()
