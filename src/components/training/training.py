from src.components.training.model_train import model_training
from src.components.training.model_valid import model_validation

from src.utils import save_as_json,load_obj,load_checkpoints,save_checkpoints,save_obj
from torch.utils.data import DataLoader
import torch
from src.logger import logger
from src.exception.exception import ExceptionNetwork,sys
import mlflow
from src.entity.config_entity import TrainingConfig
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import dagshub
dagshub.init(repo_owner='umitaktas', repo_name='VQ-Vae_Project_From_Scratch', mlflow=True)


class Training():
    
    def __init__(self,config:TrainingConfig,TEST_MODE:bool=False):
        try:
            self.config=config
            self.TEST_MODE=TEST_MODE
            
            self.encoder=load_obj(self.config.encoder_model_path).to(self.config.device)
            self.decoder= load_obj(self.config.decoder_model_path).to(self.config.device)
            self.latent_space= load_obj(self.config.latent_space_path).to(self.config.device)
            
            self.optimizer=torch.optim.Adam(params=(list(self.encoder.parameters()) + 
                                                    list(self.latent_space.parameters()) + 
                                                    list(self.decoder.parameters())
                                                    ),
                                            lr=self.config.learning_rate,
                                            betas=(self.config.beta1,self.config.beta2)
                                            )
        except Exception as e:
            ExceptionNetwork(e,sys)  
             
    def load_object(self):
        try:
            train_dataset=load_obj(self.config.train_dataset_path)
            train_dataloader=DataLoader(train_dataset,batch_size=self.config.batch_size,shuffle=True)
            
            valid_dataset=load_obj(self.config.valid_dataset_path)
            valid_dataloader=DataLoader(valid_dataset,batch_size=self.config.batch_size,shuffle=False)
            
            loss_fn=torch.nn.MSELoss(reduction="sum")        
            
            return train_dataloader,valid_dataloader,loss_fn
        
        except Exception as e:
            ExceptionNetwork(e,sys)
    
    def load_checkpoints(self,load):
        starting_epoch=1
        if load==True:
            
            starting_epoch=load_checkpoints(path=self.config.checkpoint_path,
                                            model_decoder=self.decoder,
                                            model_encoder=self.encoder,
                                            model_latent=self.latent_space,
                                            optimizer=self.optimizer)
            
            logger.info(f"Checkpoints were loaded. Training is starting from {starting_epoch}.epoch")
            
        return starting_epoch
    
    def initiate_training(self):
        try:
            result_list=[]
            
            train_dataloader,valid_dataloader,loss_fn=self.load_object()
            starting_epoch=self.load_checkpoints(load=self.config.load_checkpoint)
            
            epochs=self.config.epochs
            for epoch in range(starting_epoch,epochs):
                    
                train_loss = model_training(train_dataloader=train_dataloader,
                                            optimizer=self.optimizer,
                                            loss_fn=loss_fn,
                                            encoder=self.encoder,
                                            decoder=self.decoder,
                                            latent_space=self.latent_space,
                                            device=self.config.device)
                    
                valid_loss,real_image,predict_image = model_validation(valid_dataloader=valid_dataloader,
                                                                        loss_fn=loss_fn,
                                                                        encoder=self.encoder,
                                                                        decoder=self.decoder,
                                                                        latent_space=self.latent_space,
                                                                        device=self.config.device)
                                
                    
                save_checkpoints(save_path=self.config.checkpoint_path,
                                 model_encoder=self.encoder,
                                 model_decoder=self.decoder,
                                 model_latent=self.latent_space,
                                 optimizer=self.optimizer,epoch=epoch)
                
                logger.info(f"The last checkpoints was saved on [{self.config.checkpoint_path} ] for {epoch}.epoch")
                    
                metrics={"train_loss":train_loss,
                         "valid_loss": valid_loss,
                         "Epoch":epoch}
                    
                # save the metrics to the list
                result_list.append(metrics)

                # save the metrics to the mlflow
                if epoch%5==0:
                  mlflow.log_metrics(metrics=metrics,step=epoch)
                  mlflow.log_image(image=real_image,key="real_img",step=epoch)
                  mlflow.log_image(image=predict_image,key="predict_img",step=epoch)
                
            # save results
            save_as_json(data=result_list,save_path=self.config.results_save_path)
            logger.info(f"Training results were saved as json file on [{self.config.results_save_path} ]")
                
            # save final model
            save_obj(self.encoder,save_path=self.config.final_model_encoder_save_path)
            logger.info(f"Final model is saved on [{self.config.final_model_encoder_save_path}]")
            
            # save final model
            save_obj(self.decoder,save_path=self.config.final_model_decoder_save_path)
            logger.info(f"Final model is saved on [{self.config.final_model_decoder_save_path}]")
            
            # save final model
            save_obj(self.latent_space,save_path=self.config.final_model_latent_save_path)
            logger.info(f"Final model is saved on [{self.config.final_model_latent_save_path}]")
                
        except Exception as e:
            ExceptionNetwork(e,sys)
        
    def start_training_with_mlflow(self):
        
        try:
            
           # uri for mlflow track url in dagshub or local host
            uri="https://dagshub.com/umitaktas/VQ-Vae_Project_From_Scratch.mlflow"   # for dagshub
          
            # mlflow ui and other apps dont overlap
            # subprocess.Popen(["mlflow","ui"])
            
            # MLFLOW tracking
            mlflow.set_tracking_uri(uri=uri)
            logger.info(f"MLflow was tracked on [{uri} ]")
            

            # create a new MLFLOW experiment            
            mlflow.set_experiment("MLFLOW MyFirstExperiment")

            params={"Batch_size":self.config.batch_size,
                    "Learning_rate":self.config.learning_rate,
                    "Betas":(self.config.beta1,self.config.beta2),
                    "Epoch":self.config.epochs}
            
            # start an MLFLOW run
            with mlflow.start_run():

                # log the hyperparameters (epoch,lr,vs)
                mlflow.log_params(params=params)
                
                # Set a tag that we can use to remind ourselves what this run was for
                mlflow.set_tag("Pytorch Training Info","Environment image classification training")
                
                # Training
                self.initiate_training()
                
                logger.info("Training is completed. Metrics, parameters and model was saved on MLflow")
                        
        except Exception as e:
            ExceptionNetwork(e,sys)
            
                     
                        
            
        