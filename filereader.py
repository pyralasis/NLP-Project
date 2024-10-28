import json, requests

# takes in a url and returns the json in the file
def getFileFromUrl(url):
    response = requests.get(url)
    data = json.loads(response.text)
    return data


# A representation of a piece of data from the datasets
class DataPoint:
    def __init__(self, json_data):
        self.sent_id: str = json_data['sent_id']
        self.text: str = json_data['text']
        self.opinions: Opinion = []

        for current_opinion in json_data['opinions']:
            self.opinions.append(Opinion(current_opinion))
        
        self.vocab_size = len(self.text.split())


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


