from pkg.exception import CustomException
from components.dataTransformation import dataTransformation
from components.dataIngestion import dataIngestion
from components.model_trainer import modelTrainer

if __name__ == 'main':
    dataIngest = dataIngestion.initDataIgestion()
    df = dataIngest()
    dataTransformer = dataTransformation()
    dfEncoded = dataTransformer.initDataTransformation()
    modelTrainr = modelTrainer()
    accuracyReport,modelReport = modelTrainr.initiateModelTrainer(dfEncoded)

    print(accuracyReport)