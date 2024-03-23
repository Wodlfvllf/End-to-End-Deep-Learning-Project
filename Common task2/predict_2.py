from model_2 import *
from utils_2 import *
from dataset_2 import *

class Predict():
    '''
    Arguments :
    -> model = Trained model
    '''
    def __init__(self, model):
        super(Predict, self).__init__()
        self.model = model 

    def predict(self,x):
        preds  = self.model(x)
        return preds 