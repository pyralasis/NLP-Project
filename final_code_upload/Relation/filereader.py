import json, requests

#read in local json
def load_json_data(file_path):
    """Loads data from a JSON file and returns the list of entries."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

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
        self.rawOpinions = []
        for current_opinion in json_data['opinions']:
            self.opinions.append(Opinion(current_opinion))
            self.rawOpinions.append(current_opinion)

class Opinion:
    def __init__(self, opinion_data):
        self.source = opinion_data['Source']
        self.target = opinion_data['Target']
        self.polar_expression = opinion_data['Polar_expression']
        self.polarity = opinion_data['Polarity']
        self.intensity = opinion_data['Intensity']
