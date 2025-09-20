from src.logger import logging
from src.data.data_ingestion import DataIngestion
from src.data.data_preprocessing import DataPreprocessing
from src.features.feature_engineering import FeatureEngineer
from src.model.model_building_main import ModelBuilding
from src.model.model_evaluation import ModelEvaluation
from src.model.register_model import ModelRegister
import os   
if __name__ == "__main__":
    logging.info("The execution has started")

    try:
        # # 1. Data ingestion
        # data_ingestion = DataIngestion()
        # train_data, test_data = data_ingestion.data_complete()

        # # 2. Data preprocessing (no CSV save, in-memory)
        # data_prepro = DataPreprocessing()
        # train_processed_data, test_processed_data = data_prepro.preprocessing(train_data, test_data)

        # # 3. Feature engineering (directly from DataFrame)
        # feature_en = FeatureEngineer()
        # train_df, test_df = feature_en.feature_eng(train_processed_data, test_processed_data)


        # 1. Data ingestion
        DataIngestion.data_complete()
        # data_ingestion = DataIngestion()
        # train, test = data_ingestion.data_complete()
        

        # 2. Data preprocessing
        DataPreprocessing.preprocessing()
        # data_prepro = DataPreprocessing()
        # train_processed_data, test_processed_data = data_prepro.preprocessing(train, test)



        # 3. Feature engineering
        FeatureEngineer.feature_eng()
        # feature_en = FeatureEngineer()
        # train_df, test_df = feature_en.feature_eng(train_processed_data, test_processed_data)

        # 4. Model building
        ModelBuilding.best_model()
        # model_build = ModelBuilding()
        # model_build.best_model(train_df, test_df)


    
        # 5. Model evaluation
        # model_evaluation.evaluation_complete()
        ModelEvaluation.run_pipeline()
        # logging.info("Model evaluation completed")
        

        
        # 6. Model registration
        ModelRegister.register_complete()

    except Exception as e:
        logging.error(f"Custom Exception: {e}")
