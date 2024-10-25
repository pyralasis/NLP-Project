from filereader import *
from trainingmodels import *


def main():
    model = NeuralNetwork().to(device)
    print(model)
    
if __name__=="__main__":
    main()