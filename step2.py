import os
import argparse
import random
import math
import pickle

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-path', required=True)
parser.add_argument('-s', '--split', required=True, type=float)
parser.add_argument('-r', '--random-seed', required=True, type=int)
parser.add_argument('-o', '--output-dir', required=True)
args = parser.parse_args()
random.seed(args.random_seed)
df = pd.read_parquet(args.input_path)
# Remove non dataset sample, new added users
df_original = df[df.id <= 60]
speaker_count = len(df_original.id.unique())
test_count = math.floor(args.split * speaker_count)
test_ids = random.sample(list(df_original.id.unique()), k=test_count)
print(f'Sample users saved for test: {test_ids}')
df_train = df_original[~df.id.isin(test_ids)]
print(f'Training set size {len(df_train)}')
df_test = df_original[df.id.isin(test_ids)]
print(f'Test set size {len(df_test)}')

# Note: only training set is used to create min max
# This prevents test data interfering with training
min_max_dict = {}
for column in df_train.columns:
    if column != 'id':
        min_max = {
            'min': df_train[column].min(),
            'max': df_train[column].max(),
        }
        min_max_dict[column] = min_max
print(min_max_dict)

def apply(df, min_max):
    for column in df.columns:
        if column != 'id':
            min_v = min_max[column]['min']
            max_v = min_max[column]['max']
            df[column] = (df[column] - min_v) / (max_v - min_v)

apply(df_train, min_max_dict)
apply(df_test, min_max_dict)
train_path = os.path.join(args.output_dir, 'train.parquet')
df_train.to_parquet(train_path)
test_path = os.path.join(args.output_dir, 'test.parquet')
df_test.to_parquet(test_path)
min_max_path = os.path.join(args.output_dir, 'minmax.pickle')
with open(min_max_path, 'wb') as f:
    pickle.dump(min_max_dict, f)
