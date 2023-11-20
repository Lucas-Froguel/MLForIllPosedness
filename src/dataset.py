
import pymongo
import pandas as pd
from torch.utils.data import Dataset, DataLoader

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
        # There are 10**8 points
        # self.collection.count_documents({"kernel": self.kernel})
        return 99000

    def __getitem__(self, idx):
        query = {
            "idx": idx,
            "kernel": self.kernel
        }

        data = self.collection.find_one(query, projection={"_id": False})

        return data["k"], data["integral"]
    
    def close_connection(self):
        self.client.close()


def load_dataset(
    database: str = None,
    collection: str = None,
    db_url: str = None,
    kernel: str = None,
    batch_size: int = None,
    generator = None
) -> DataLoader:
    kernel_data = CustomImageDataset(
        kernel=kernel, database=database, collection=collection, db_url=db_url
    )
    kernel_loader = DataLoader(kernel_data, batch_size=batch_size, shuffle=True, generator=generator)

    return kernel_loader
