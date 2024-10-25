from filereader import *


def main():
    data = getFileFromUrl('https://raw.githubusercontent.com/jerbarnes/semeval22_structured_sentiment/refs/heads/master/data/opener_en/dev.json')
    myDataPoints = []
    for item in data:
        myDataPoints.append(dataPoint(item))
    
if __name__=="__main__":
    main()