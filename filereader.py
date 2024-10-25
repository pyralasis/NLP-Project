import json, requests

# takes in a url and returns the json in the file
def getFileFromUrl(url):
    response = requests.get(url)
    data = json.loads(response.text)
    return data
