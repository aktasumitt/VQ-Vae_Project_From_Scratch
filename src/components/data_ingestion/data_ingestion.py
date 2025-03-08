from src.logger import logger
from src.exception.exception import ExceptionNetwork, sys
from src.entity.config_entity import DataIngestionConfig
from pathlib import Path
from src.utils import save_as_json


class DataIngestion():
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    # Loading Dataset
    def loading_Dataset(self,dataset_dir):
        
        img_path_list=[]
        
        for i,folder in enumerate(Path(dataset_dir).glob("*")):
            for _,img in enumerate(Path(folder).glob("*")):
                img_path_list.append(str(img))
            
            if i==3: break
                
        logger.info(f"...{len(img_path_list)} data were loaded")    
        return img_path_list

    def initiate_data_ingestion(self):
        
        image_path_list=self.loading_Dataset(dataset_dir=self.config.local_data_path)
        save_as_json(image_path_list,save_path=self.config.save_data_path)
        logger.info(f"data path list was saved on {self.config.save_data_path}")
        

if __name__=="__main__":
    
    
    data_ingestion=DataIngestion()
    data_ingestion.initiate_data_ingestion()