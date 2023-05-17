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
import keras
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
from sklearn.metrics import classification_report, confusion_matrix
import os
import fnmatch
import collections
import base64
import cv2
import face_recognition
import os
import pickle
import hashlib

app = Flask(__name__)
CORS(app)
blockchain = bc.Blockchain()
blockchain.create_genesis_block()

peers = set()

##################################################
# Blockchain
##################################################

@app.route('/register_node', methods=['POST'])
def register_new_node():
    node_addr = request.get_json()["node_address"]
    if not node_addr:
        return "Invalid data", 400
    peers.add(node_addr)
    return get_chain()

@app.route('/register_node_with', methods=['POST'])
def register_with():
    node_addr = request.get_json()["node_address"]
    if not node_addr:
        return "Invalid data", 400
    data = {"node_address": request.host_url}
    headers = {"Content-Type": "application/json"}

    response = requests.post(node_addr + "/register_node", data=json.dumps(data), headers=headers)
    print(response)
    if response.status_code != 200:
        print("mal")
        return response.content, response.status_code
    else:
        global blockchain
        global peers
        chain_dump=response.json()['chain']
        peers.update(response.json()['peers'])
        blockchain=create_chain_from_dup(chain_dump)
        print(blockchain)
        return "Registration succesful", 200

def create_chain_from_dup(chain_dump):
    new_blockchain=bc.Blockchain()
    for id, data in enumerate(chain_dump):
        block=bc.block.Block(data["index"], data["transactions"], data["timestamp"], data["previous_block"], data["nonce"])
        proof = data['hash']
        if id>0:
            added = new_blockchain.add_block(block, proof)
            if not added:
                raise Exception("The chain dump is tampered!")
        else:
            block.hash=block.compute_hash()
            new_blockchain.chain.append(block)
    return new_blockchain

@app.route('/add_block', methods=['POST'])
def verify_and_add_block():
    data=json.loads(request.get_json())
    block = bc.block.Block(data["index"], data["transactions"], data["timestamp"], data["previous_block"], data['nonce'])
    proof = data['hash']
    print(data)
    added = blockchain.add_block(block, proof)
    if not added:
        return "The block was discarded by the node", 400
    return "Block added to the chain", 201


def announce_new_block(block):
    for peer in peers:
        print(block.__dict__)
        url = "{}add_block".format(peer)
        print(url)
        requests.post(url, json=json.dumps(block.__dict__, sort_keys=True)) 
        
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

############################################################
#Emergency
############################################################

@app.route('/emergencies', methods=['POST'])
def emergencies():
    data=request.get_json()
    require_fields = ["device", "user"]
    for field in require_fields:
        if not data.get(field):
            return "Invalid transaction data", 404
    s=cur.execute("SELECT rol from users where email='"+data.get("user")+"'")
    user_p=s.fetchone()[0]
    s=cur.execute("SELECT permission from devices where device_id='"+str(data.get("device"))+"'")
    device_p=s.fetchone()[0]
    if(user_p=="admin"):
        permission=device_p
    elif(device_p=="admin"):
        permission=user_p
    else:
        if(user_p==device_p):
             permission=user_p
        else:
            permission=""
    locations=cur.execute("SELECT * from emergency where permission='"+permission+"'")
    print(locations)
    return jsonify(list(locations))

############################################################
#Voice recognition
############################################################

def preProcessData(csvFileName):
    print(csvFileName+ " will be preprocessed")
    info = pd.read_csv(csvFileName)
    filenameArray = info['filename'] 
    speakerArray = []
    #print(filenameArray)
    for i in range(len(filenameArray)):
        speaker = int(filenameArray[i].split("_")[0].split("r")[1])
        speakerArray.append(speaker)
    info['number'] = speakerArray
    #Dropping unnecessary columns
    info = info.drop(['filename'],axis=1)
    info = info.drop(['label'],axis=1)
    info = info.drop(['chroma_stft'],axis=1)
    info.shape

    print("Preprocessing is finished")
    print(info.head())
    return info

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
    #y_test=np.array(y_test, dtype=int)
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
    s=cur.execute("SELECT user_id from users where email='"+mail+"'")
    id=s.fetchone()[0]
    filename="Speaker"+str(id).zfill(4)+"_000.wav"
    f = open('../new_audios/test_file/'+filename, 'wb')
    f.write(bytearray(data['data']))
    f.close()
    return jsonify({"Res": predict_audio(filename, "../new_audios/test_file/")})

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
count={0: 73, 1: 93, 2: 95, 3: 67, 4: 71, 5: 56, 6: 59, 7: 48, 8: 78, 9: 51, 10: 42, 11: 51, 12: 26, 13: 28, 14: 39, 15: 29, 16: 37, 17: 34, 18: 31, 19: 25, 20: 399, 21: 45, 22: 399, 23: 30, 24: 42, 25: 38, 26: 35, 27: 36, 28: 46, 29: 23, 30: 25, 31: 36, 32: 28, 33: 27, 34: 26, 35: 24, 36: 25, 37: 43, 38: 26, 39: 39, 40: 24, 41: 24, 42: 31, 43: 26, 44: 28, 45: 32, 46: 29, 47: 26, 48: 23, 49: 39, 50: 25, 51: 23, 52: 23, 53: 23, 54: 399, 55: 7, 56: 7, 57: 7, 58: 7, 59: 7, 60: 7}

@app.route('/voice_register', methods=['POST'])
def register():
    mail=request.get_json()['email']
    data=request.get_json()['data']
    print(mail)
    shutil.rmtree('../new_audios/train2/')
    os.mkdir('../new_audios/train2/')
    cur.execute("INSERT INTO users (email, rol) VALUES ('"+mail+"', 'admin')")
    con.commit()
    s=cur.execute("SELECT user_id from users where email='"+mail+"'")
    id=s.fetchone()[0]
    print(id)
    for x in data:
        f=open('../new_audios/train2/Speaker'+str(id).zfill(4)+"_"+str(x).zfill(3)+".wav", "wb")
        f.write(bytearray(data[str(x)]['data']))

    NEW_USER = "../new_user.csv"
    extractWavFeaturesmult("../new_audios/train2", NEW_USER)
    train_data = preProcessData(TRAIN_CSV_FILE)
    newData = preProcessData(NEW_USER)
    #dataFrame = pd.read_csv(NEW_USER)
    #dataFrame.to_csv(TRAIN_CSV_FILE, mode='a', index=False, header=False)
    trainData=train_data.append(newData, ignore_index=True)
    # Splitting the dataset into training, validation and testing dataset
    X = np.array(trainData.iloc[:, :-1], dtype = float)
    y = trainData.iloc[:, -1]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=50)
    #utilizar antic scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform( X_train )
    X_val = scaler.transform( X_val )
    count[id]=1-20/399
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
    es = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)
    y_train=np.array(y_train, dtype=int)
    print("hola2")
    y_val=np.array(y_val, dtype=int)
    history = model.fit(X_train,y_train,validation_data=(X_val, y_val),epochs=100,batch_size=128, class_weight=count, callbacks=[es])    
    joblib.dump(scaler, '../scaler.save')
    model.save('../speaker-recognition.h5')
    return {"Res":"OK"}

############################################
# Behavioral recognition
############################################
def predict(test):
    b_model = keras.models.load_model('../behavior/behavior-recognition.h5')
    b_scaler = joblib.load('../behavior/behavior_scaler.save') 
    X_test = np.array(test.iloc[:, :-1], dtype = float)
    y_test = test.iloc[:, -1]
    X_test = b_scaler.transform( X_test )   
    score = b_model.evaluate(X_test, y_test)
    return score[1]

@app.route('/behavior_recognition', methods=['POST'])
def behavior_recognition():
    obj=request.get_json()
    array=np.array(obj)
    test=pd.DataFrame([x for i, x in enumerate(array) if i!=0], columns=array[0].split(','))
    test=test.drop(columns='time_secs')
    return jsonify({"Res": predict(test)})


###################################################
# Face recognition
###################################################
@app.route('/face_recognition', methods=['POST'])
def face():
    mail=request.get_json()['email']
    data=request.get_json()['data']
    count=0
    s=cur.execute("SELECT salt from users where email='"+mail+"'")
    sa=s.fetchone()[0]
    salt=bytes.fromhex(sa)
    plaintext=mail.encode()
    digest=hashlib.pbkdf2_hmac('sha256', plaintext, salt, 10000)
    hex_hash=digest.hex()
    f=open('users/'+str(hex_hash)+'.txt', 'rb')
    info=pickle.load(f)
    f.close()
    for x in data:
        new_data=x.replace('data:image/png;base64,', '')
        d=np.frombuffer(base64.b64decode(new_data), dtype=np.uint8)
        img=cv2.imdecode(d, flags=1)
        face=detect_face2(img)
        if type(face) != bool:
            result = face_recognition.compare_faces(info, face)
            for x in result:
                if x==True:
                    count=count+1
    print("Percentatge: "+str(count)+" %")
    if(count>360): 
        val=True
    else:
        val=False
    return jsonify({"Res": val})

@app.route('/face_register', methods=['POST'])
def face_register():
    mail=request.get_json()['email']
    salt=os.urandom(16)
    plaintext=mail.encode()
    digest=hashlib.pbkdf2_hmac('sha256', plaintext, salt, 10000)
    hex_hash=digest.hex()
    cur.execute("""INSERT INTO users (email,salt, rol) VALUES (?,?,?)""", (mail, salt.hex(), 'admin'))
    con.commit()
    data=request.get_json()['data']
    info=[]
    for x in data:
        new_data=x.replace('data:image/png;base64,', '')
        d=np.frombuffer(base64.b64decode(new_data), dtype=np.uint8)
        img=cv2.imdecode(d, flags=1)
        face=detect_face2(img)
        if type(face) != bool:
            info.append(face)
    f=open('users/'+str(hex_hash)+'.txt', 'wb')
    pickle.dump(info, f)
    f.close()
    with open('test.png', 'wb') as f:
        f.write(img)
    return {'Res': 'OK'}

def detect_face2(image):
    face_loc=face_recognition.face_locations(image)
    if len(face_loc)==0:
        return False
    else:
        face_loc=face_loc[0]
    return face_recognition.face_encodings(image, known_face_locations=[face_loc])[0]


app.run(debug=True, host='0.0.0.0', port=9000)

