import json, requests

# takes in a url and returns the json in the file
def getFileFromUrl(url):
    response = requests.get(url)
    data = json.loads(response.text)
    return data


class DataPoint:
    def __init__(self, json_data):
        self.sent_id = json_data['sent_id']
        self.text = json_data['text']
        self.opinions = []
        for current_opinion in json_data['opinions']:
            self.opinions.append(Opinion(current_opinion))

class Opinion:
    def __init__(self, opinion_data):
        self.source = opinion_data['Source']
        self.target = opinion_data['Target']
        self.polar_expression = opinion_data['Polar_expression']
        self.polarity = opinion_data['Polarity']
        self.intensity = opinion_data['Intensity']
    
    # MIGHT NEED LATER
    def to_tuple(self):
        ...
