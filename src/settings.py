from pathlib import Path
import decouple

BASE_DIR = Path(__file__).resolve().parent.parent

config = decouple.AutoConfig(BASE_DIR)


MONGO_DATABASE_URL = config("MONGO_DATABASE_URL")
MONGODB_NAME = config("MONGO_DATABASE")
MONGO_DATA_COLLECTION=config("MONGO_DATA_COLLECTION")

kernels = dict()
