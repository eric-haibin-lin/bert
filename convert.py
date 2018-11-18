from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import pickle

import numpy as np
import mxnet as mx
from tensorflow.python import pywrap_tensorflow

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, required=True,
                    help="Checkpoint filename. For example, "
                         "cased_L-12_H-768_A-12/bert_model.ckpt")
parser.add_argument('--show_converted', action='store_true',
                    help='Display converted parameter names')

args = parser.parse_args()
file_name = args.file_name

def read_tf_checkpoint(name):
    tensors = {}
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
        tensors[key] = reader.get_tensor(key)
    return tensors

MAPPING = [
  ("bert/encoder/layer_", "encoder.transformer_cells."),
  ("/attention/self/", ".attention_cell."),
  ("key", "proj_key"),
  ("query", "proj_query"),
  ("value", "proj_value"),
  ("/attention/output/LayerNorm/", ".layer_norm."),
  ("/attention/output/dense/", ".proj."),
  ("cls/seq_relationship/output_weights", "classifier.weight"),
  ("cls/seq_relationship/output_bias", "classifier.bias"),
  ("cls/predictions/output_bias", "decoder.3.bias"),
  ("cls/predictions/transform/dense/","decoder.0."),
  ("cls/predictions/transform/LayerNorm/","decoder.2."),
  ("kernel", "weight"),
  ("/intermediate/dense/", ".ffn.ffn_1."),
  ("/output/dense/", ".ffn.ffn_2."),
  ("/output/LayerNorm/", ".ffn.layer_norm."),
  ("bert/embeddings/LayerNorm/", "encoder.layer_norm."),
  ("bert/embeddings/position_embeddings", "encoder.position_weight"),
  ("bert/embeddings/token_type_embeddings", "token_type_embed.0.weight"),
  ("bert/embeddings/word_embeddings", "word_embed.0.weight"),
  ("bert/pooler/dense/", "pooler."),
  ("/","."),
]

tensors = read_tf_checkpoint(file_name)
names = sorted(tensors.keys())

converted = {}
for source_name in names:
    source, source_t = tensors[source_name], tensors[source_name].T
    target, target_name = source, source_name
    for old, new in MAPPING:
        target_name = target_name.replace(old, new)
    if 'kernel' in source_name:
        target = source_t
    converted[target_name] = target
    if source_t.shape == source.shape and len(source.shape) > 1 and target is not source_t:
        print('warning misleading shape', target_name, target.shape)
    if args.show_converted:
        print('%s: %s'%(target_name, target.shape))
    else:
        print('%s: %s'%(source_name, source.shape))
print("Total number of parameters: ", len(tensors))
print("Total number of converted parameters: ", len(converted))
with open(file_name + '.converted', 'wb') as f:
    pickle.dump(converted, f, pickle.HIGHEST_PROTOCOL)
print("Converted parameters stored at %s"%(file_name+'.converted'))
