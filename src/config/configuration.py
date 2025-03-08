from src.entity.config_entity import (TestConfig,
                                      TrainingConfig,
                                      DataIngestionConfig,
                                      ModelIngestionConfig,
                                      DataTransformationConfig,
                                      PredictionConfig)

from src.constants.config import Config
from src.constants.params import Params


class Configuration():

    def __init__(self):

        self.config = Config
        self.params = Params

    def data_ingestion_config(self):

        configuration = DataIngestionConfig(local_data_path=self.config.local_data_path,
                                            save_data_path=self.config.save_data_path)

        return configuration

    def data_transformation_config(self):

        configuration = DataTransformationConfig(train_dataset_save_path=self.config.train_dataset_save_path,
                                                 valid_dataset_save_path=self.config.valid_dataset_save_path,
                                                 test_dataset_save_path=self.config.test_dataset_save_path,
                                                 test_rate=self.params.test_rate,
                                                 valid_rate=self.params.valid_rate,
                                                 img_path_list_path=self.config.save_data_path)

        return configuration

    def model_config(self):

        configuration = ModelIngestionConfig(encoder_save_path=self.config.encoder_save_path,
                                             decoder_save_path=self.config.decoder_save_path,
                                             latent_space_save_path=self.config.latent_space_save_path,
                                             num_embedding=self.params.num_embedding,
                                             hidden_dim=self.params.hidden_dim,
                                             channel_size=self.params.channel_size)

        return configuration

    def training_config(self):

        configuration = TrainingConfig(encoder_model_path=self.config.encoder_save_path,
                                       decoder_model_path=self.config.decoder_save_path,
                                       latent_space_path=self.config.latent_space_save_path,
                                       train_dataset_path=self.config.train_dataset_save_path,
                                       valid_dataset_path=self.config.valid_dataset_save_path,
                                       checkpoint_path=self.config.checkpoint_save_path,
                                       final_model_encoder_save_path=self.config.final_model_encoder_save_path,
                                       final_model_decoder_save_path=self.config.final_model_decoder_save_path,
                                       final_model_latent_save_path=self.config.final_model_latent_save_path,
                                       results_save_path=self.config.results_save_path,
                                       batch_size=self.params.batch_size,
                                       device=self.params.device,
                                       learning_rate=self.params.learning_rate,
                                       beta1=self.params.beta1,
                                       beta2=self.params.beta2,
                                       epochs=self.params.epochs,
                                       load_checkpoint=self.params.load_checkpoint
                                       )

        return configuration

    def test_config(self):

        configuration = TestConfig(final_model_encoder_path=self.config.final_model_encoder_save_path,
                                   final_model_decoder_path=self.config.final_model_decoder_save_path,
                                   final_model_latent_path=self.config.final_model_latent_save_path,
                                   test_dataset_path=self.config.test_dataset_save_path,
                                   batch_size=self.params.batch_size,
                                   load_checkpoints_for_test=self.params.load_checkpoints_for_test,
                                   checkpoint_path=self.config.checkpoint_save_path,
                                   test_result_save_path=self.config.test_result_save_path,
                                   test_real_image_path=self.config.test_real_img_save_path,
                                   test_generated_image_path=self.config.test_generated_img_path,
                                   test_model_encoder_save_path=self.config.test_model_encoder_save_path,
                                   test_model_decoder_save_path=self.config.test_model_decoder_save_path,
                                   test_model_latent_save_path=self.config.test_model_latent_save_path,
                                   save_tested_model=self.params.save_tested_model
                                   )

        return configuration

    def prediction_config(self):

        configuration = PredictionConfig(img_size=self.params.img_size,
                                         encoder_path=self.config.final_model_encoder_save_path,
                                         latent_path=self.config.final_model_latent_save_path,
                                         decoder_path=self.config.final_model_decoder_save_path,
                                         predicted_img_save_path=self.config.predicted_img_save_path,
                                         prediction_img_load_path=self.config.prediction_img_load_path                                        
                                         )

        return configuration
