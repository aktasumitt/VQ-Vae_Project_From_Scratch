import PIL.Image
from src.entity.config_entity import TestConfig
from src.components.testing.test_module import model_test
from src.utils import save_as_json,load_obj,load_checkpoints,save_obj

from torch.utils.data import DataLoader
import torch
from PIL import Image
from src.logger import logger
from src.exception.exception import ExceptionNetwork,sys

class Testing():
    def __init__(self,config:TestConfig):
        self.config=config
        
        self.encoder=load_obj(self.config.final_model_encoder_path)
        self.decoder= load_obj(self.config.final_model_decoder_path)
        self.latent_space= load_obj(self.config.final_model_latent_path)
        

    def load_object(self):
        try:
            test_dataset=load_obj(self.config.test_dataset_path)
            test_dataloader=DataLoader(test_dataset,batch_size=self.config.batch_size,shuffle=True)

            loss_fn=torch.nn.CrossEntropyLoss()        
            
            return test_dataloader,loss_fn
        
        except Exception as e:
            raise ExceptionNetwork(e,sys)
            
    def initiate_testing(self):
        try:
            
            test_dataloader,loss_fn = self.load_object()
            
            # Load Checkpoints if u want
            if self.config.load_checkpoints_for_test==True:
                load_checkpoints(path=self.config.checkpoint_path,
                                model_decoder=self.decoder,
                                model_encoder=self.encoder,
                                model_latent=self.latent_space)    
            
            test_loss, real_image,predict_image = model_test(test_dataloader=test_dataloader,
                                                            loss_fn=loss_fn,
                                                            encoder=self.encoder,
                                                            decoder=self.decoder,
                                                            latent_space=self.latent_space)
                            
            save_as_json(data={"Test_loss":test_loss},save_path=self.config.test_result_save_path)
            real_image.save(self.config.test_real_image_path)
            predict_image.save(self.config.test_generated_image_path)
            logger.info(f"Testing model is completed. Test results was saved")
            
            
            
            # Save tested model if you want (you can use this after load spesific checkpoints)
            if self.config.save_tested_model==True:
                # save final model
                save_obj(self.encoder,save_path=self.config.test_model_encoder_save_path)
                logger.info(f"Final model is saved on [{self.config.test_model_encoder_save_path}]")
                
                # save final model
                save_obj(self.decoder,save_path=self.config.test_model_decoder_save_path)
                logger.info(f"Final model is saved on [{self.config.test_model_decoder_save_path}]")
                
                # save final model
                save_obj(self.latent_space,save_path=self.config.test_model_latent_save_path)
                logger.info(f"Final model is saved on [{self.config.test_model_latent_save_path}]")             
    
                
            
        
        except Exception as e:
            raise ExceptionNetwork(e,sys)