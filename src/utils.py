import json
import os
import yaml
import pickle
import torch
from box import ConfigBox
from pathlib import Path
from src.exception.exception import ExceptionNetwork,sys

def save_as_json(data,save_path:Path):
    try:
        save_dir=os.path.dirname(save_path)
        os.makedirs(save_dir,exist_ok=True)
        
        save_path=Path(save_path)
        
        with open(save_path,"w") as f:
            json.dump(data,f)
            
    except Exception as e:
        raise ExceptionNetwork(e,sys)

def load_json(path:Path):
    try:
        path=Path(path)
        with open(path, "r") as f:
            loaded_data = json.load(f)
        
        return loaded_data
    
    except Exception as e:
        raise ExceptionNetwork(e,sys)


def save_as_yaml(data,save_path:Path):
    try:
        save_dir=os.path.dirname(save_path)
        os.makedirs(save_dir,exist_ok=True)
        
        save_path=Path(save_path)
        
        with open(save_path, "w", encoding="utf-8") as file:
            yaml.dump(data, file, allow_unicode=True)
            
    except Exception as e:
        raise ExceptionNetwork(e,sys)


def load_yaml(path:Path):  
    try:     
        path=Path(path)
        with open(f"{path}", "r", encoding="utf-8") as file:
            loaded_dict = yaml.safe_load(file) 
        
        return ConfigBox(loaded_dict)
    
    except Exception as e:
        raise ExceptionNetwork(e,sys)


def save_obj(data,save_path:Path):
    try:    
        save_dir=os.path.dirname(save_path)
        os.makedirs(save_dir,exist_ok=True)
        
        save_path=Path(save_path)
        
        with open(save_path, "wb") as file:
            torch.save(data, file)
            
    except Exception as e:
        raise ExceptionNetwork(e,sys)
        
        
def load_obj(path:Path):
    try:    
        path=Path(path)
        with open(path, "rb") as file:
            obj=torch.load(file,map_location=torch.device("cpu"),weights_only=False)
            
        return obj
    
    except Exception as e:
        raise ExceptionNetwork(e,sys)


def save_checkpoints(save_path,epoch,model_encoder,model_decoder,model_latent,optimizer):
    try:    
        checkpoints_name=Path(f"checkpoint_{epoch}-epoch.pth.tar") # to save every_epoch
        
        save_dir=os.path.dirname(save_path)
        os.makedirs(save_dir,exist_ok=True)
        
        
        epoch_path=os.path.join(save_dir,checkpoints_name)
        latest_path=Path(save_path)
        
        checkpoint={"Epoch":epoch,
                    "Optimizer_State":optimizer.state_dict(),
                    "Model_encoder":model_encoder.state_dict(),
                    "Model_decoder":model_decoder.state_dict(),
                    "Model_latent":model_latent.state_dict()}
        
        torch.save(checkpoint,f=latest_path)
        torch.save(checkpoint,f=epoch_path)
        
    except Exception as e:
        raise ExceptionNetwork(e,sys)


def load_checkpoints(model_encoder,model_decoder,model_latent,path:Path,optimizer:None)->int:
    try:    
        path=Path(path)
        
        checkpoint=ConfigBox(torch.load(path))
        model_decoder.load_state_dict(checkpoint.Model_encoder)
        model_encoder.load_state_dict(checkpoint.Model_decoder)
        model_latent.load_state_dict(checkpoint.Model_latent)
        
        if optimizer != None:
            start_epoch=checkpoint.Epoch
            optimizer.load_state_dict(checkpoint.Optimizer_State)
        
            return start_epoch+1
    
    except Exception as e:
        raise ExceptionNetwork(e,sys)
    

    