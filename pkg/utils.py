import os
import sys

import numpy as np
import pandas as pd
import dill
import pickle


from pkg.exception import CustomException

def saveObj(filePath, obj):
    try:
        dirPath = os.path.dirname(filePath)
        os.makedirs(dirPath,exist_ok=True)

        with open(filePath, "wb") as fileObj:
            dill.dump(obj,fileObj)

    except Exception as e:
        raise CustomException(e,sys)
    
def loadObj(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)