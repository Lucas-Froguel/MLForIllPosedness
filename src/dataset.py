
import pymongo
import pandas as pd
from torch.utils.data import Dataset

from src.mongodb.queries.general_queries import get_one_document_query


class CustomImageDataset(Dataset):
    def __init__(
        self, kernel: str = "x1", database: str = None, collection: str = None, db_url: str = None
    ):
        self.database = database
        self.collection = collection
        self.client = pymongo.MongoClient(db_url)
        self.db = self.client[self.database]
        self.collection = self.db[self.collection]
        self.kernel = kernel

    def __len__(self):
        # this is the amount of data points, the current length method would take too long
        # self.collection.count_documents({"kernel": self.kernel})
        return 10**8

    def __getitem__(self, idx):
        query = {
            "idx": idx,
            "kernel": self.kernel
        }

        data = self.collection.find_one(query, projection={"_id": False})

        return data["k"], data["integral"]
    
    def close_connection(self):
        self.client.close()
