from asyncio import protocols
from glob import glob
import mimetypes
from sqlite3 import enable_shared_cache
from unittest import result
from urllib import response
from pkg_resources import require
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import blockchain as bc
import time
import json
import shutil
#requisits and ignore warnings
import warnings
warnings.simplefilter('ignore')
from keras import models
from keras import layers
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import os
import librosa
import pandas as pd
import csv
from sklearn import preprocessing
import joblib
import wave
import fnmatch
import collections
import keras
import librosa
import csv
import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jwt
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import models, layers
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot, cm
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix

app = Flask(__name__)
CORS(app)
blockchain = bc.Blockchain()

peers = set()

def encode_auth_token(self, device_UUID):
    try:
        payload ={
            'exp' : datetime.datetime.utcnow() + datetime.timedelta(days=0, seconds=5),
            'iat' : datetime.datetime.utcnow(),
            'sub' : device_UUID
        }
        return jwt.encode(
            payload,
            app.config.get('SECRET_KEY'),
            algorithm='HS256'
        )
    except Exception as e:
        return e

@app.route('/register_node', methods=['POST'])
def register_new_node():
    node_addr = request.get_json()["node_address"]
    if not node_addr:
        return "Invalid data", 400
    peers.add(node_addr)
    return(get_chain)

@app.route('/register_node_with', methods=['POST'])
def register_with():
    node_addr = request.get_json()["node_address"]
    if not node_addr:
        return "Invalid data", 400
    data = {"node_address": request.host_url}
    headers = {"Content-Type": "application/json"}

    response = requests.post(node_addr + "/register_node", data=json.dumps(data), headers=headers)
    if response != 200:
        return response.content, response.status_code
    else:
        return response.content, response.status_code

def create_chain_from_dup(chain_dump):
    blockchain=blockchain()
    for id, data in enumerate(chain_dump):
        block=bc.block.Block(data["index"], data["transactions"], data["timestamp"], data["previous_block"])
        proof = data['hash']
        if id>0:
            added = blockchain.add_block(block, proof)
            if not added:
                raise Exception("The chain dump is tampered!")
        else:
            blockchain.chain.append(block)
    return blockchain

@app.route('/add_block', methods=['POST'])
def verify_and_add_block():
    data=request.get_json()
    block = bc.block.Block(data["index"], data["transactions"], data["timestamp"], data["previous_block"])
    proof = data['hash']
    added = blockchain.add_block(block, proof)
    if not added:
        return "The block was discarded by the node", 400
    return "Block added to the chain", 201

def announce_new_block(block):
    for peer in peers:
        url = "{}add_block".format(peer)
        requests.post(url, data=json.dumps(block.__dict__, sort_keys=True))   

@app.route('/new_transaction', methods=['POST'])
def new_transaction():
    data=request.get_json()
    require_fields = ["data", "pk", "UUID"]
    for field in require_fields:
        if not data.get(field):
            return "Invalid transaction data", 404
    
    data["timestamp"]=time.time()
    blockchain.add_new_transaction(data)
    return "Success", 201

@app.route('/block_exists', methods=['POST'])
def block_exists():
    data=request.get_json()
    require_fields = ["data", "pk", "UUID"]
    for field in require_fields:
        if not data.get(field):
            return "Invalid transaction data", 404
    val=blockchain.block_exists(data, blockchain.chain)
    print(val)
    if(val=="Error"): 
        return "Incorrect", 401
    elif (val): 
        return "Success", 200 
    else: 
        return "Created", 201

@app.route('/get_pk', methods=['GET'])
def get_pk():
    print("/get_pk")
    UUID=request.args.get('UUID')
    val=blockchain.get_pk(UUID, blockchain.chain)
    j=encode_auth_token(UUID)
    print(j)
    if(val):
        return jsonify(val)
    else:
        return "Incorrect", 401


@app.route('/chain', methods=['GET'])
def get_chain():
    chain = []
    for block in blockchain.chain:
        chain.append(block.__dict__)
    return json.dumps({"length": len(chain), "chain":chain, "peers":list(peers)})

@app.route('/mine', methods=['GET'])
def mine():
    result=blockchain.mine()
    if not result:
        return "No blockchain to mine"
    else:
        chain_length = len(blockchain.chain)
        print(chain_length)
        concensus()
        if chain_length == len(blockchain.chain):
            announce_new_block(blockchain.last_block)
        return "Block #{} is mined".format(blockchain.last_block.index)


@app.route('/pending_transactions', methods=['GET'])
def get_pending_transactions():
    return json.dumps(blockchain.unconfirmed_transactions)

def concensus():
    #jerarquitzar
    global blockchain
    longest_chain = None
    current_len = len(blockchain.chain)
    for node in peers:
        response = requests.get('{}/chain'.format(node))
        lenght = response.json()['lenght']
        chain = response.json()['chain']
        if lenght > current_len and blockchain.check_chain_validity(chain):
            current_len = lenght
            longest_chain = chain
    
    if longest_chain:
        blockchain=longest_chain
        return True

    return False

con = sqlite3.connect('tfm.db', check_same_thread=False)
cur = con.cursor()


#Voice recognition

def preProcessData(csvFileName):
    print(csvFileName)
    data = pd.read_csv(csvFileName)
    print(data.head())
    filenameArray = data['filename'] 
    speakerArray = []
    #print(filenameArray)
    for i in range(len(filenameArray)):
        speaker = int(filenameArray[i].split("_")[0].split("r")[1])
        speakerArray.append(speaker)
    data['number'] = speakerArray
    #Dropping unnecessary columns
    data = data.drop(['filename'],axis=1)
    data = data.drop(['label'],axis=1)
    data = data.drop(['chroma_stft'],axis=1)
    data.shape
    return data

def getSpeaker(speaker):
    speaker = "Speaker"+str(speaker).zfill(3)
    return speaker
    
        
def printPrediction(X_data, y_data, printDigit, model):
    print('\n# Generate predictions')
    for i in range(len(y_data)):
        predict_x=model.predict(X_data[i:i+1])[0]
        predict_classes = np.argmax(predict_x)
        prediction = getSpeaker(predict_classes)
    
        speaker = getSpeaker(y_data[i])
        if printDigit == True:
           print("Number={0:d}, y={1:10s}- prediction={2:10s}- match={3}".format(i, speaker, prediction, speaker==prediction))
        else:
           print("y={0:10s}- prediction={1:10s}- match={2}".format(speaker, prediction, speaker==prediction))
           if(speaker==prediction): 
            return True
           else: 
            return False

def extractWavFeatures(filename, dirname, csvFileName):
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()
    file = open(csvFileName, 'w', newline='')
    #with file:
    writer = csv.writer(file)
    writer.writerow(header)
    number = f'{dirname}{filename}'
    y, sr = librosa.load(number, mono=True, duration=30)
    # remove leading and trailing silence
    y, index = librosa.effects.trim(y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    writer.writerow(to_append.split())
    file.close()

def predict_audio(filename, dirname):
    new_model = keras.models.load_model('../speaker-recognition.h5')
    scaler = joblib.load('../scaler.save') 
    extractWavFeatures(filename, dirname, final_test)
    final_testData = preProcessData(final_test)
    X_test = np.array(final_testData.iloc[:, :-1], dtype = float)
    y_test = final_testData.iloc[:, -1]
    X_test = scaler.transform( X_test )
    y_test=np.array(y_test, dtype=int)
    print(set(y_test))
    score = new_model.evaluate(X_test, y_test)
    # Prediction
    return printPrediction(X_test, y_test, False, new_model)
    

final_test="../final_test.csv"
new_model = keras.models.load_model('../speaker-recognition.h5')
scaler = joblib.load('../scaler.save') 

@app.route('/voice_recognition', methods=['POST'])
def recognition():
    mail=request.get_json()['email']
    data=request.get_json()['data']
    print(mail)
    s=cur.execute("SELECT user_id from users where email='"+mail+"'")
    id=s.fetchone()[0]
    filename="Speaker"+str(id).zfill(4)+"_000.wav"
    print(filename)
    new_model = keras.models.load_model('../speaker-recognition.h5')
    scaler = joblib.load('../scaler.save') 
    f = open('../new_audios/test_file/'+filename, 'wb')
    f.write(bytearray(data['data']))
    f.close()
    return jsonify({"Res": predict_audio(filename, "../new_audios/test_file/")})

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import librosa
import csv
import os
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')

def extractWavFeaturesmult(soundFilesFolder, csvFileName):
    print("The features of the files in the folder "+soundFilesFolder+" will be saved to "+csvFileName)
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()
    print('CSV Header: ', header)
    file = open(csvFileName, 'w', newline='')
    #with file:
    writer = csv.writer(file)
    writer.writerow(header)
    for filename in tqdm(os.listdir(soundFilesFolder)):
        number = f'{soundFilesFolder}/{filename}'
        y, sr = librosa.load(number, mono=True, duration=30)
        # remove leading and trailing silence
        y, index = librosa.effects.trim(y)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        writer.writerow(to_append.split())
    file.close()
    print("End of extractWavFeatures")
    
TRAIN_CSV_FILE = "../train.csv"

@app.route('/voice_register', methods=['POST'])
def register():
    mail=request.get_json()['email']
    data=request.get_json()['data']
    #print(data['0']['data'])
    shutil.rmtree('../new_audios/train2/')
    os.mkdir('../new_audios/train2/')
    cur.execute("INSERT INTO users (email) VALUES ('"+mail+"')")
    con.commit()
    s=cur.execute("SELECT user_id from users where email='"+mail+"'")
    id=s.fetchone()[0]
    for x in data:
        f=open('../new_audios/train2/Speaker'+str(id).zfill(4)+"_"+str(x).zfill(3)+".wav", "wb")
        f.write(bytearray(data[str(x)]['data']))

    NEW_USER = "../new_user.csv"
    extractWavFeaturesmult("../new_audios/train2", NEW_USER)
    dataFrame = pd.read_csv(NEW_USER)
    dataFrame.to_csv(TRAIN_CSV_FILE, mode='a', index=False, header=False)
    trainData = preProcessData(TRAIN_CSV_FILE)
    #testData = preProcessData(TEST_CSV_FILE)
    # Splitting the dataset into training, validation and testing dataset
    X = np.array(trainData.iloc[:, :-1], dtype = float)
    y = trainData.iloc[:, -1]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=50)
    scaler = StandardScaler()
    X_train = scaler.fit_transform( X_train )
    X_val = scaler.transform( X_val )
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(len(y_train), activation='softmax'))
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    y_train=np.array(y_train, dtype=int)
    y_val=np.array(y_val, dtype=int)
    history = model.fit(X_train,y_train,validation_data=(X_val, y_val),epochs=100,batch_size=1, callbacks=[es])
    joblib.dump(scaler, '../scaler.save') 
    model.save('../speaker-recognition.h5')
    return {"Res":"OK"}


app.run(debug=True, port=8000)