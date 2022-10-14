from asyncio import protocols
from glob import glob
from sqlite3 import enable_shared_cache
from unittest import result
from urllib import response
from pkg_resources import require
from flask import Flask, request
from flask_cors import CORS
import requests
import blockchain as bc
import time
import json

app = Flask(__name__)
CORS(app)
blockchain = bc.Blockchain()

peers = set()

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


app.run(debug=True, port=8000)