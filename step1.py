import argparse
import os
import re

import numpy as np
import pandas as pd
import librosa

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--speaker-files', required=True)
parser.add_argument('-o', '--output', required=True)
args = parser.parse_args()

dir_pattern = re.compile('Speaker([0-9]+)')


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


df = []
for speaker_directory in os.listdir(args.speaker_files):
    speaker_path = os.path.join(args.speaker_files, speaker_directory)
    found = re.search(dir_pattern, speaker_directory)
    if os.path.isdir(speaker_path) and found is not None:
        sample = {'id': int(found.group(1))}
        for sample_file in os.listdir(speaker_path):
            try:
                sample_path = os.path.join(speaker_path, sample_file)
                df.append(get_features(sample_path, sample))
            except Exception as e:
                print(f'error handling {sample_path} {e}')
pd.DataFrame(df).to_parquet(args.output)
