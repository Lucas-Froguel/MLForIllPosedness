from src.exceptions import DatabaseError, ItemNotFound
from src.mongodb.mongodb import MongoConnection


def get_one_document_query(
    database: str = None,
    collection: str = None,
    query: dict = None,
    projection: dict = {},
    convert_id: bool = False,
    raise_exception: bool = True
):
    with MongoConnection() as mongo:
        col = mongo.get_db_collection(database, collection)
        item = col.find_one(query, projection=projection)
        if item:
            if convert_id:
                item["_id"] = str(item["_id"])
            return item
        if raise_exception:
            raise ItemNotFound(details=str(query))


def insert_one_document_query(
    database: str = None, collection: str = None, data: dict = None
):
    try:
        with MongoConnection() as mongo:
            obj = mongo.insert_document(
                db_name=database,
                collection_name=collection,
                document=data
            )
            return str(obj.inserted_id)
    except Exception as e:
        raise DatabaseError(details=str(e))


def bulk_insert_documents_query(
    database: str = None, collection: str = None, data: list = None
):
    try:
        with MongoConnection() as mongo:
            mongo.bulk_insert_documents(
                db_name=database,
                collection_name=collection,
                documents=data
            )
    except Exception as e:
        raise DatabaseError(details=str(e))
