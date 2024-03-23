from Model_c1 import *
from utils_c1 import *
from dataset_c1 import *

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