from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    local_data_path:Path 
    save_data_path:Path
    

@dataclass
class DataTransformationConfig:
    img_path_list_path: Path
    train_dataset_save_path : Path 
    valid_dataset_save_path : Path
    test_dataset_save_path : Path
    test_rate: float
    valid_rate: float
    
    
    
@dataclass
class ModelIngestionConfig:
    encoder_save_path: Path 
    decoder_save_path: Path 
    latent_space_save_path: Path 
    channel_size: int
    num_embedding: int
    hidden_dim: int

    
@dataclass
class TrainingConfig:
    encoder_model_path:Path
    decoder_model_path:Path
    latent_space_path:Path
    train_dataset_path: Path 
    valid_dataset_path:Path
    checkpoint_path: Path 
    final_model_encoder_save_path: Path
    final_model_decoder_save_path: Path 
    final_model_latent_save_path: Path
    results_save_path: Path 
    batch_size: int 
    device: str 
    learning_rate: float 
    beta1: float
    beta2: float 
    epochs: int
    load_checkpoint: bool
 
    
@dataclass
class TestConfig:
    final_model_encoder_path: Path 
    final_model_decoder_path: Path 
    final_model_latent_path: Path 
    test_dataset_path:Path
    batch_size:int
    load_checkpoints_for_test:bool
    checkpoint_path:Path
    test_result_save_path:Path
    test_real_image_path:Path
    test_generated_image_path:Path
    test_model_encoder_save_path:Path
    test_model_decoder_save_path:Path
    test_model_latent_save_path:Path
    save_tested_model:bool


@dataclass
class PredictionConfig:
    img_size: int 
    encoder_path: Path 
    latent_path: Path 
    decoder_path: Path
    predicted_img_save_path: Path
    prediction_img_load_path : Path
