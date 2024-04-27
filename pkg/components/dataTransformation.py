import os
import sys
from pkg.config import Config
from pkg.exception import CustomException
from pkg.logger import logging
#from src.utils import saveObj



import pandas as pd
import numpy as np

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

@dataclass
class dataTransformationConfig:
    preprocessorObjFilePath = os.path.join('artifacts',"preprocessor.pkl")

class dataTransformation:
    def __init__(self):
        #self.dataTransformationConfig = dataTransformationConfig()
        self.target = Config.target
        self.features = Config.features

    def getDataTransformerObject(self):
        pass

        
    def initiateDataTransformation(self,trainPath,testPath=None):

        try:
            traindf = pd.read_csv(trainPath)
            traindf = traindf[self.features]

            numericalFeatures = traindf.select_dtypes(include=['int64','float64']).columns.to_list()
            categoricalFeatures = traindf.select_dtypes(include=['object']).columns.to_list()

            catOrdCols = ['EDUCATION']
            catNomCols = ['MARITALSTATUS','GENDER','last_prod_enq2','first_prod_enq2']

            #Ordinal Encoding of Oridinal features
            eduUniq = traindf['EDUCATION'].unique().tolist()
            traindf['EDUCATION'] = traindf['EDUCATION'].replace(eduUniq,[2,3,1,4,3,1,3])

            # One Hot Encoding
            traindfEncoded = pd.get_dummies(traindf,columns=catNomCols,dtype='int64')

            logging.info('Applied Transformations')

            return traindfEncoded



        except Exception as e:
            raise CustomException(e,sys)
        