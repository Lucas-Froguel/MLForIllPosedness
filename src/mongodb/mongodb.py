
import pymongo
from src import settings
from dataclasses import dataclass

from src.exceptions import DatabaseError


@dataclass
class MongoConnection:
    url_db: str = settings.MONGO_DATABASE_URL

    def __enter__(self):
        self.client = pymongo.MongoClient(self.url_db)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def get_db_status(self, db_name: str = None):

        """Method that gets the server status"""
        try:
            return self.client[db_name].command("serverStatus")

        except Exception as e:
            raise DatabaseError(details=str(e))

    def get_db_collections(self, db_name: str = None):

        """Method that get all database collections (tables)"""

        try:
            return list(self.client[db_name].list_collection_names())

        except Exception as e:
            raise DatabaseError(details=str(e))

    def get_db_collection(self, db_name: str = None, collection_name: str = None):
        """Method that returns a single collection"""
        try:
            db = self.client[db_name]
            collection = db[collection_name]
        except Exception as e:
            raise DatabaseError(details=str(e))
        return collection

    def get_all_collection_documents(
        self, db_name: str = None, collection_name: str = None
    ):

        """Method that get all documents from a collections (tables)"""

        try:
            db = self.client[db_name]
            collection = db[collection_name]
            documents = [doc for doc in collection.find()]

            return documents

        except Exception as e:
            raise DatabaseError(details=str(e))


    def insert_document(
        self, db_name: str = None, collection_name: str = None, document: dict = []
    ):
        """Method that inserts one document with id into a collection"""

        db = self.client[db_name]
        collection = db[collection_name]
        obj = collection.insert_one(document)
        return obj

    def bulk_insert_documents(
        self, db_name: str = None, collection_name: str = None, documents: list = []
    ):
        """Method that inserts many documents with id into a collection"""

        db = self.client[db_name]
        collection = db[collection_name]
        collection.insert_many(documents)

    def update_data(
        self,
        db_name: str = None,
        collection_name: str = None,
        filter: dict = [],
        newvalues: dict = [],
    ):
        """Method that update data into the db"""

        db = self.client[db_name]
        collection = db[collection_name]
        obj = collection.update_one(filter, newvalues)
        return obj
