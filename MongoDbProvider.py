import pymongo
import datetime
import json
import os
import pandas as pd

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
            query = { "time": { "$gt" : startDt, "$lt" : endDt }, 'hasEvaluated':False }
        else:
            query = {
                        "CompanyName" : companyName,
                        "time" : { "$gt" : startDt, "$lt" : endDt },
                        'hasEvaluated': False
                    }
        cursor = collection.find(query)
        contents = []

        for doc in cursor:
            contents.append( ( doc['_id'], doc['body'] ) )

        return contents

    def MarkANewsHasEvaluated(self, newsIds):

        for newsId in newsIds:
            self._collection.update_one({ '_id': newsId }, { '$set':{'HasEvaluated':True}} )

    def InsertAnalyzedResult(self, companyName, eDt, result):
        arCollection = self._database['AnalyzedResult']
        arCollection.insert_one( { 'companyName': companyName, 'endDateTime': eDt, 'results': result } )

    def ReadAnalyzedResultsAsDataFrame(self):
        arCollection = self._database['AnalyzedResult']
        cursor = arCollection.find({});

        data = []
        for doc in cursor:
            companyName = doc["companyName"]
            eDt = doc['endDateTime']
            result = doc['results']
            data.append([companyName, eDt] + result)

        return pd.DataFrame(data, columns=['companyName', 'Date', 'sent', 'sent_norm', 'sent_PN', 'sent_norm_std', 'sent_PN_std'])

    def ReadFromQueryParamsFile(self):

        filePath = os.getcwd() + '\\QueryParams.json'
        with open(filePath) as json_file:
            data = json.load(json_file)
        return data

