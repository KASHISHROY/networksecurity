import pymongo
import os
from dotenv import load_dotenv

load_dotenv()

client = pymongo.MongoClient(os.getenv("MONGO_DB_URL"))

db = client["KASHISHROY"]
collection = db["NetworkData"]

print("Total docs:", collection.count_documents({}))
print("One sample:", collection.find_one())