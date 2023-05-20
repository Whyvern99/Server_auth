import argparse

import pandas as pd
import tensorflow as tf
import numpy as np
import random
from keras.callbacks import EarlyStopping

# To many warning, I know what I am doing
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-data', required=True)
parser.add_argument('-r', '--test-data', required=True)
parser.add_argument('-o', '--output-model', required=True)
args = parser.parse_args()

df = pd.read_parquet(args.input_data)
df_t = pd.read_parquet(args.test_data)
# Num of features - label col
input_shape = (len(df.columns) - 1,)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

def extract_class_weights(labels):
    pos = sum(labels)
    total = len(labels)
    neg = total - pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    return {0: weight_for_0, 1: weight_for_1}
    
list=df.id.unique()

es = EarlyStopping(monitor='val_accuracy', patience=20, verbose=1)

for _ in range(10):
    print('Starting training')
    model.summary()
    random.shuffle(list)
    for user_id in list:
        data = np.array(df.drop('id', axis=1), dtype=np.float32)
        labels = np.array(df.id == user_id, dtype=np.uint32)
        class_weight = extract_class_weights(labels)

        # Reset last layer weights as we are only interested on metalearning
        # Only the previous layers will help on new users
        last_layer = model.layers[-1]
        weight_initializer = last_layer.kernel_initializer
        bias_initializer = last_layer.bias_initializer
        old_weights, old_biases = last_layer.get_weights()
        last_layer.set_weights([
            weight_initializer(shape=old_weights.shape),
            bias_initializer(shape=old_biases.shape)
        ])

        model.fit(
            x=data,
            y=labels,
            validation_split=0.3,
            shuffle=True,
            class_weight=class_weight,
            verbose=0,
            epochs=10
        )

    print('#' * 40)
    print(' Evaluation ')
    print('#' * 40)
    # Model evaluation
    for user_id in df_t.id.unique():
        print(f'Evaluation with speaker: {user_id}')
        data = np.array(df_t.drop('id', axis=1), dtype=np.float32)
        labels = np.array(df_t.id == user_id, dtype=np.uint32)

        # Only last layer should be used for new subject
        model_t = tf.keras.models.clone_model(model)
        model_t.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        model_t.set_weights(model.get_weights())
        for layer in model_t.layers[:-1]:
            layer.trainable = False

        class_weight = extract_class_weights(labels)
        print(class_weight)

        model_t.fit(
            x=data,
            y=labels,
            validation_split=0.5,
            shuffle=True,
            class_weight=class_weight,
            epochs=10
        )


model.save(args.output_model)
