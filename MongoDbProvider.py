import pymongo
import datetime
import json
import os

class DatabaseProvider:

    def __init__(self, host="127.0.0.1", port=27017, dbName="news"):
        self._dbName = dbName
        self._client = pymongo.MongoClient(host, port)
        self._database = self._client[dbName]
        self._collection = self._database["Combined"]

    def __del__(self):
        # Close and release database resources
        self._client.close()

    def ReadNewsFromDatabase(self, companyName = "", startDt:datetime = datetime.datetime(2015,1,1), endDt:datetime = datetime.datetime.now()):

        collection = self._collection

        ## Replay the startDt and endDt with IOSDate('{dateString}') while test the query in MongoCampass

        if companyName == "":
            query = { "time": { "$gt" : startDt, "$lt" : endDt } }
        else:
            query = {
                        "CompanyName" : companyName,
                        "time" : { "$gt" : startDt, "$lt" : endDt }
                    }
        cursor = collection.find(query)
        contents = []

        for doc in cursor:
            contents.append(doc['body'])

        return contents


    def ReadFromQueryParamsFile(self):

        filePath = os.getcwd() + '\\QueryParams.json'
        with open(filePath) as json_file:
            data = json.load(json_file)
        return data

