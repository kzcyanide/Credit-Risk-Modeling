import os
import sys


import pandas as pd
import numpy as np

from dataclasses import dataclass


from pkg.config import Config
from pkg.exception import CustomException
from pkg.logger import logging
from pkg.utils import saveObj

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

@dataclass
class modelTrainerConfig():
    trainedModelFilePath = os.path.join("models","model.pkl")

class modelTrainer():
    def __init__(self):
        self.modelTrainerConfig = modelTrainerConfig()

    def trainModels(self,df,models,features,target,drop_cols=None):

        y = df[target]
        X = df.drop(target,axis=1)
        
        fold = range(0,Config.n_splits)
        
        score_types = ['Accuracy']+[f'Precision Class {i}' for i in range(0, 4)] + \
            [f'Recall Class {i}' for i in range(0, 4)] + \
            [f'F1 Score Class {i}' for i in range(0, 4)]
        
        multiindex = pd.MultiIndex.from_product([fold,score_types],names=['Fold','Metric'])
        
        FtreImp = pd.DataFrame(columns = models, 
                            index   = [c for c in X.columns ]
                           ).fillna(0)
        
        cv = StratifiedKFold(n_splits=Config.n_splits,shuffle=True,random_state=42)
        
        OOF_Preds = pd.DataFrame()
        Scores = pd.DataFrame(index=multiindex,columns=models)
        accuracyScore = pd.DataFrame(index=range(Config.n_splits),columns=models)

        for fold,(train_idx,valid_idx) in enumerate(cv.split(X,y)):
            Xtrain = X.iloc[train_idx]
            Xvalid = X.iloc[valid_idx]
            ytrain = y[train_idx]
            yvalid = y[valid_idx]
            print(f'fold {fold}')

            for mdl in models.keys():
                model = models[mdl]
                
                model.fit(Xtrain,ytrain)

                    # Collating feature importance:-
                try:
                    FtreImp[model] += model.feature_importances_
                except: 
                    pass

                validPreds = model.predict(Xvalid)
                trainPreds = model.predict(Xtrain)
                precision,recall,f1_score,_ = precision_recall_fscore_support(yvalid,validPreds)
                accuracy = accuracy_score(yvalid,validPreds)
                Scores.loc[fold,'Accuracy'][mdl] = accuracy
                accuracyScore.loc[fold][mdl] = accuracy
                

                for i in range(0,4):
                    Scores.loc[fold,f'Precision Class {i}'][mdl] = precision[i]
                    Scores.loc[fold,f'Recall Class {i}'][mdl] = recall[i]
                    Scores.loc[fold,f'F1 Score Class {i}'][mdl] = f1_score[i]
                
        return accuracy_score,Scores


    def initiateModelTrainer(self,traindf,test_arr = None):
        try:
            logging.info("Splitting train and test input data")

            models = {'XGB':XGBClassifier(**Config.xgb_params),
                      'CB':CatBoostClassifier(**Config.cb_params)
                      }
            drop_cols = []

            accuracyReport,modelReport = self.trainModels(traindf,Config.features,Config.target)



            bestModel = max(accuracyReport.mean().to_dict())

            bestModel = models[bestModel]
            bestScore = models[bestModel]

            if bestScore < 0.6:
                raise CustomException("No best model found")
            
            logging.info("Best found model on both training and testing dataset")

            saveObj(
                filePath=self.modelTrainerConfig.trainedModelFilePath,
                obj=bestModel
            )

            logging.info("Model Training Completed")

            #predicted = bestModel.predict(X_test)
            #r2Score = r2_score(y_test,predicted)

            return accuracyReport,modelReport


        except Exception as e:
            raise CustomException(e,sys)