import json

#获取APIkey
def getapikey():
    with open('..\\jsondir\\info.json') as credentials_file:
        credentials = json.load(credentials_file)
    key=credentials['key']
    return key
