# k-fold cross-validation
import numpy as np
import pandas as pd 

from sklearn import model_selection

def create_folds(data):
    # We create a new column called kfold and fill it with -1
    data["kfold"] = -1

    # The next step is to randomize the rows of the data
    data = data.sample(frac=1).reset_index(drop=True)