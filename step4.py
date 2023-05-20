import re
import numpy as np
import pandas as pd
import librosa
import os
import argparse
import random
import math
import pickle

def get_features(path, base_dict):
    y, sr = librosa.load(path, mono=True)
    y, index = librosa.effects.trim(y)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_dict = {f'mfcc{idx + 1}': np.mean(mfcc) for idx, mfcc in enumerate(mfccs)}
    return {
        'chroma_stft': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        'rmse': np.mean(librosa.feature.rms(y=y)),
        'spec_cent': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'spec_bw': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        'rolloff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        'zcr': np.mean(librosa.feature.zero_crossing_rate(y)),
    } | mfcc_dict | base_dict


# ###df = []
# sample = {'id': 61}
# speaker_path="/home/silvia/Escritorio/tfm/Server_auth/audios/audios/data/Speaker0061"
# for sample_file in os.listdir(speaker_path):
#     try:
#         sample_path = os.path.join(speaker_path, sample_file)
#         df.append(get_features(sample_path, sample))
#     except Exception as e:
#         print(f'error handling {sample_path} {e}')
# pd.DataFrame(df).to_parquet('new_user.parquet')

with open("output/minmax.pickle", 'rb') as f:
    min_max_dict=pickle.load(f)

def apply(df, min_max):
    for column in df.columns:
        if column != 'id':
            min_v = min_max[column]['min']
            max_v = min_max[column]['max']
            df[column] = (df[column] - min_v) / (max_v - min_v)

df = pd.read_parquet("data.parquet")
# Remove non dataset sample, new added users
df_original = df[df.id <= 61]
test_ids = [61]
print(f'Sample users saved for test: {test_ids}')

df_train = df_original[~df.id.isin(test_ids)]
print(f'Training set size {len(df_train)}')
df_test = df_original[df.id.isin(test_ids)]
print(f'Test set size {len(df_test)}')

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