import pymongo
import datetime
import json
import os

class DatabaseProvider:

    def __init__(self, host="172.16.1.105", port=27017, dbName="factiva"):
        self._dbName = dbName
        self._client = pymongo.MongoClient(host, port)
        self._database = self._client[dbName]
        self._collection = self._database["News"]

    def __del__(self):
        # Close and release database resources
        self._client.close()

    def ReadNewsFromDatabase(self, companyName = "", startDt:datetime = datetime.datetime(2015,1,1), endDt:datetime = datetime.datetime.now()):

        collection = self._collection

        if companyName == "":
            query = { "DateTime": { "$gt" : startDt, "$lt" : endDt } }
        else:
            query = {
                        "COMPANY" : companyName,
                        "DateTime" : { "$gt" : startDt, "$lt" : endDt }
                    }
        cursor = collection.find(query)
        contents = []

        for doc in cursor:
            contents.append(doc['CONTENT'])

        return contents


    def ReadFromQueryParamsFile(self):

        filePath = os.getcwd() + '\\QueryParams.json'
        with open(filePath) as json_file:
            data = json.load(json_file)
        return data

