import pandas as pd
import numpy as np

from pkg.exception import CustomException
from components.dataTransformation import dataTransformation
from components.dataIngestion import dataIngestion
from components.model_trainer import modelTrainer

if __name__ == '__main__':
    df = pd.read_csv('./input/train.csv')
    dataTransformer = dataTransformation()
    dfEncoded = dataTransformer.initiateDataTransformation(df)
    modelTrainr = modelTrainer()
    accuracyReport,modelReport = modelTrainr.initiateModelTrainer(dfEncoded)

    print(accuracyReport)