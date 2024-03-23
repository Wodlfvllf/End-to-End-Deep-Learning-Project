from Model_mae import *
from utils_mae import *
from dataset_mae import *

class Predict():
    '''
    Arguments :
    -> model = Trained model
    '''
    def __init__(self, model):
        super(Predict, self).__init__()
        self.model = model 

    def predict(self,x):
        _, preds, _  = self.model(x)
        return preds