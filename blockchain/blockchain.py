from ast import Break
from unittest import result
import block
import time
from pprint import pprint
from hashlib import sha256
from rsa import compute_hash
import json



class Blockchain:

    dif=2

    def __init__(self):
        self.chain = []
        self.unconfirmed_transactions = []
        self.create_genesis_block() 
    
    def create_genesis_block(self):
        genesis_block=block.Block(0, [], time.time(), "0")
        genesis_block.hash=genesis_block.compute_hash()
        self.chain.append(genesis_block)

    @property
    def last_block(self):
        return self.chain[-1]

    def proof_of_work(self, block):
        block.nonce=0
        computed_hash=block.compute_hash()
        while not computed_hash.startswith('0' * Blockchain.dif):
            block.nonce += 1
            computed_hash=block.compute_hash()
        return computed_hash
    
    def add_block(self, block, proof):
        previous_block=self.last_block.hash
        if previous_block!=block.previous_block:
            return False
        if not self.is_valid_proof(block, proof):
            return False
        block.hash = proof
        self.chain.append(block)
        return True
    
    def is_valid_proof(self, block, block_hash):
        return (block_hash.startswith('0' * Blockchain.dif) and block_hash == block.compute_hash())

    def add_new_transaction(self, transaction):
        self.unconfirmed_transactions.append(transaction)
    
    def mine(self):
        if not self.unconfirmed_transactions:
            return False
        last_block = self.last_block
        index=int(last_block.index) + 1
        new_block = block.Block(index=index, transactions=self.unconfirmed_transactions, timestamp=time.time(), previous_block=last_block.hash)
        proof = self.proof_of_work(new_block)
        added=self.add_block(new_block, proof)
        if not added:
            return False

        """
        Si es una lista no se debería quitar simplemente el primer elemento? El indice es incremental?
        """
        self.unconfirmed_transactions = []
        return new_block.index
    
    def check_chain_validity(cls, chain):
        result = True
        previous_block = "0"
        for block in chain:
            hash=block.hash
            delattr(block, hash)

            if not cls.is_valid_proof(block, block.hash) or previous_block!=block.previous_block:
                result = False
                Break
            
            block.hash, previous_block = hash, hash
        return result

    def block_exists(cls, data, chain):
        for block in chain:
            transaction=block.transactions
            if transaction and data["UUID"]==transaction[0]['UUID']:
                    if data['pk']==transaction[0]['pk'] and data['data']==transaction[0]['data']:
                        return True
                    else:
                        return 'Error'
                    break
        cls.add_new_transaction(data)
        cls.mine()
        return False
