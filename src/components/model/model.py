from src.components.model.encoder import Encoder
from src.components.model.decoder import Decoder
from src.components.model.embedding import Latent_Space
from src.exception.exception import ExceptionNetwork,sys
from src.utils import save_obj
from src.logger import logger
from src.entity.config_entity import ModelIngestionConfig


class ModelIngestion():
    
    def __init__(self, config: ModelIngestionConfig):
        try:
            self.config = config
            
            self.encoder=Encoder(config.channel_size,config.hidden_dim) # Encoder Model
            
            self.latent_space=Latent_Space(num_embedding=config.num_embedding) # Latent space (embedding)
            
            self.decoder=Decoder(config.channel_size,config.hidden_dim) # Decoder Model
            
        except Exception as e:
            raise ExceptionNetwork(e,sys)
        
    def initiate_and_save_model(self):
        try:
            save_obj(self.encoder, self.config.encoder_save_path)
            save_obj(self.decoder, self.config.decoder_save_path)
            save_obj(self.latent_space, self.config.latent_space_save_path)
            
            logger.info("encoder, decoder ve latent_space modelleri artifacts içerisine kaydedildi")
            
        except Exception as e:
            raise ExceptionNetwork(e,sys)
        
if __name__ == "__main__":
    config = ModelIngestionConfig()
    model_ingestion = ModelIngestion(config)
    model_ingestion.initiate_and_save_model()

