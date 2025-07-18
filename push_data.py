import json 
import os 
import sys 
from dotenv import load_dotenv


load_dotenv()

MONGODB_URL = os.getenv("MONGODB_CLUSTER_URL")
# print(MONGODB_URL)

import certifi
ca = certifi.where()

import pandas as pd 
import numpy as np 
import pymongo
from network_sec.exception.exception import NetworkSecurityException
from network_sec.logs.logger import logging
import certifi

class NetworkDataExtraction():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def csv_to_json(self, file_path):
        try:
            df = pd.read_csv(file_path)
            df.reset_index(drop=True, inplace=True)
            record = list(json.loads(df.T.to_json()).values())
            return record
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        

    def push_data(self, database, collection, records):
        try:
            print(MONGODB_URL)
            self.database = database
            self.collection = collection
            self.records = records

            print("self-loaded")
            self.mongoclient = pymongo.MongoClient(
                MONGODB_URL
            )

            print("connection_build")
            self.database = self.mongoclient[self.database]
            self.collection = self.database[self.collection]
            print("insertion_time")
            self.collection.insert_many(records)

            return len(self.records)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

        

if __name__ =="__main__":
    FILE_PATH = os.path.join("network_data", "raw_data", "data.csv")
    DATABASE = "NetworkSecurity"
    Collection = "NetworkData"

    dataextract = NetworkDataExtraction()
    records = dataextract.csv_to_json(FILE_PATH)
    print(records)
    ok = dataextract.push_data(DATABASE, Collection, records)
    print(ok)



    


    
        

