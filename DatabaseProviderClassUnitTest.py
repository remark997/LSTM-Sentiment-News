import json
from MongoDbProvider import DatabaseProvider
import datetime

mdb = DatabaseProvider()
queryList = mdb.ReadFromQueryParamsFile()

#result = mdb.ReadNewsFromDatabase('3M Co', datetime.datetime(1990,3,1), datetime.datetime(2000,1,1))

for queryIndex in range(len(queryList)):
    print(queryList[queryIndex]["companyName"], datetime.datetime.strptime(queryList[queryIndex]["sDt"], "%Y-%m-%dT00:00:00Z"), datetime.datetime.strptime(queryList[queryIndex]["eDt"],"%Y-%m-%dT00:00:00Z"))
    result = mdb.ReadNewsFromDatabase(queryList[queryIndex]["companyName"], datetime.datetime.strptime(queryList[queryIndex]["sDt"], "%Y-%m-%dT%H:%M:%SZ"), datetime.datetime.strptime(queryList[queryIndex]["eDt"],"%Y-%m-%dT%H:%M:%SZ"))
    print(result)





