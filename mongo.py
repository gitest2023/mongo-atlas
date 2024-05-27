from glob import glob
from math import ceil
import os
from pathlib import Path
from random import choices
import re

import cv2
import matplotlib.pyplot as plt
from PIL import Image

from pymongo import MongoClient
from pymongo.errors import CollectionInvalid, DuplicateKeyError
from pymongo.operations import SearchIndexModel

from sentence_transformers import SentenceTransformer
from tqdm.notebook import tqdm

MONGODB_URI= 'mongodb+srv://mongo_dev:1234maiSON@cluster0.6jloogy.mongodb.net/'
DATABASE_NAME = "fingerprint"
IMAGE_COLLECTION_NAME = "fingerprint"

class Mongo:

    client = MongoClient(MONGODB_URI)
    db = client.get_database(DATABASE_NAME)

    # Load CLIP model.
    # This may print out warnings, which can be ignored.
    model = SentenceTransformer("clip-ViT-L-14")

    @classmethod
    def get_model(cls):
        return cls.model

    @classmethod
    def create_collection(cls, collection_name: str=''):
        # Ensure the collection exists, because otherwise you can't add a search index to it.
        try:
            cls.db.create_collection(collection_name)
            print('A new collection has already been created.')
        except CollectionInvalid:
            print("Images collection already existed: " + collection_name)

    @classmethod
    def get_collection(cls, name: str=IMAGE_COLLECTION_NAME):
        return cls.db.get_collection(name)

    @classmethod
    def insert_one(cls, doc: dict={}):
        collection = cls.get_collection()
        try:
            collection.insert_one(doc)
        except DuplicateKeyError as e:
            print(e)

    @classmethod
    def create_index(cls, name: str='default', collection_name: str=IMAGE_COLLECTION_NAME):
        # Add a search index (if it doesn't already exist):
        collection = cls.get_collection(name=collection_name)
        if len(list(collection.list_search_indexes(name=name))) == 0:
            print("Creating search index...")
            # Define fields:
            # {
            #     "fields": [
            #         {
            #         "type": "vector",
            #         "path": "embedding",
            #         "numDimensions": 769,
            #         "similarity": "cosine"
            #         }
            #     ]
            # }

            collection.create_search_index(
                SearchIndexModel(
                    definition={
                        "mappings": {
                            "dynamic": True,
                            "fields": {
                                "embedding": {
                                    "dimensions": 768,
                                    "similarity": "cosine",
                                    # "type": "knnVector",
                                    "type": "vector",
                                }
                            },
                        }
                    },
                    name=name,
                )
            )

            # collection.create_search_index(
            #     {
            #         "definition": {
            #             "mappings": {
            #                 "dynamic": True,
            #                 "fields": {
            #                     'embedding' : {
            #                         "dimensions": 1536,
            #                         "similarity": "dotProduct",
            #                         "type": "knnVector"
            #                     }
            #                 }
            #             }
            #         },
            #         "name": name
            #     }
            # )
            print("Done.")
        else:
            print("Vector search index already exists")

    @classmethod
    def search_image(cls, image, thresh: float=0.99):
        """
        Use MongoDB Vector Search to search for a matching image.

        The `search_phrase` is first converted to a vector embedding using
        the `model` loaded earlier in the Jupyter notebook. The vector is then used
        to search MongoDB for matching images.
        """
        collection = cls.get_collection()
        embedding = cls.model.encode(image)
        cursor = collection.aggregate(
            [
                {
                    "$vectorSearch": {
                        # "index": "default",
                        "index": "fingerprint01",
                        "path": "embedding",
                        "queryVector": embedding.tolist(),
                        "numCandidates": 100,
                        "limit": 9,
                    }
                },
                {"$project": {"_id": 1, "score": {"$meta": "vectorSearchScore"}}},
            ]
        )
        return [ x for x in list(cursor) if x['score'] >= thresh]
        # return list(filter(lambda x: x['score'] >= thresh, list(cursor)))


# Load CLIP model.
# This may print out warnings, which can be ignored.
# model = SentenceTransformer("clip-ViT-L-14")

def load_images(image_count=1000):
    """
    Load `image_count` images into the database, creating an embedding for each using the sentence transformer above.

    This can take some time to run if image_count is large.

    The image's pixel data is not loaded into MongoDB, just the image's path and vector embedding.
    """
    image_paths = choices(glob("fingerprints/*.jpg", recursive=True), k=image_count)
    collection = Mongo.get_collection()
    model = Mongo.get_model()
    for path in tqdm(image_paths):
        emb = model.encode(Image.open(path))
        try:
            print('path => ', path)
            collection.insert_one(
                {
                    "_id": re.sub("fingerprints/", "", path),
                    "embedding": emb.tolist(),
                }
            )
        except DuplicateKeyError as e:
            print(e)

def display_images(docs, cols=3, show_paths=False):
    """
    Helper function to display some images in a grid.
    """
    for doc in docs:
        doc["image_path"] = "images/" + doc["_id"]

    rows = ceil(len(docs) / cols)

    f, axarr = plt.subplots(nrows=rows, ncols=cols, figsize=(8, 8), tight_layout=True)
    for i, doc in enumerate(docs):
        image_path = doc["image_path"]
        score = doc["score"]
        image = cv2.imread(image_path)[:, :, ::-1]
        axis = axarr[i // cols, i % cols]
        axis.imshow(image)
        axis.axis("off")
        if show_paths:
            axis.set_title(image_path.rsplit("/", 1)[1])
        else:
            axis.set_title(f"Score: {score:.4f}")
    plt.show()




if __name__ == '__main__':
    # create_collection(IMAGE_COLLECTION_NAME)
    # create_index('fingerprint01')
    # load_images(12)
    # img = Image.open('images/test-6-052098011479-l.jpg')
    img = Image.open('fingerprints/2-004171001789-l.jpg')
    print(Mongo.search_image(img))
    # create_index('fingerprint01')