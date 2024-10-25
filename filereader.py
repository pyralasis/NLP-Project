import json, requests

# takes in a url and returns the json in the file
def getFileFromUrl(url):
    response = requests.get(url)
    data = json.loads(response.text)
    return data


class dataPoint:
    def __init__(self, jsonData):
        self.sentId = jsonData['sent_id']
        self.text = jsonData['text']
        self.opinions = []
        for currentOpinion in jsonData['opinions']:
            self.opinions.append(opinion(currentOpinion))

class opinion:
    def __init__(self, opinionData):
        self.source = opinionData['Source']
        self.target = opinionData['Target']
        self.polarExpression = opinionData['Polar_expression']
        self.polarity = opinionData['Polarity']
        self.intensity = opinionData['Intensity']
