import json
from hashlib import sha256

class Block:
    def __init__(self, index, transactions, timestamp, previous_block, nonce=0):
        """
        index: Unique ID
        transactions: data
        timestamp: time of generation of the block
        """
        self.index=index
        self.transactions=transactions
        self.timestamp=timestamp
        self.previous_block=previous_block
        self.nonce = nonce
    
    def compute_hash(self):
        """
        Returns the hash of a block
        """
        block_string=json.dumps(self.__dict__, sort_keys=True)
        return sha256(block_string.encode()).hexdigest()