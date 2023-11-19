import pandas as pd 

from src.mongodb.queries.general_queries import bulk_insert_documents_query
from src.settings import MONGODB_NAME, MONGO_DATA_COLLECTION


features = pd.read_csv("src/data/integrals_feature.csv")
labels = pd.read_csv("src/data/integrals_label.csv")

print("Data loaded, starting upload...")

for col in features:
    print("Loading column...")
    data = []
    t = 0
    for ind, item in enumerate(labels[col]):
        integral_data = {
            "idx": ind,
            "kernel": col,
            "integral": item,
            "k": features["x1"][ind]
        }
        data.append(integral_data)
        t += 1
        if t == 1000:
            print("Inserting 1k...")
            bulk_insert_documents_query(database=MONGODB_NAME, collection=MONGO_DATA_COLLECTION, data=data)
            data = []
            t = 0

