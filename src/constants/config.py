class Config:
    # Data Ingestion Stage
    local_data_path = "local_data/128X128imgs"
    save_data_path = "artifacts/data_ingestion/data_paths.json"

    # Data Transformation Stage
    train_dataset_save_path = "artifacts/data_transformation/train_dataset.pth"
    valid_dataset_save_path = "artifacts/data_transformation/valid_dataset.pth"
    test_dataset_save_path = "artifacts/data_transformation/test_dataset.pth"

    # Model Ingestion Stage
    encoder_save_path = "artifacts/model/encoder.pth"
    decoder_save_path = "artifacts/model/decoder.pth"
    latent_space_save_path = "artifacts/model/latent_space.pth"

    # Training Stage
    checkpoint_save_path = "callbacks/checkpoints/checkpoint_latest.pth.tar"
    final_model_encoder_save_path = "callbacks/final_model/encoder.pth"
    final_model_decoder_save_path = "callbacks/final_model/decoder_model.pth"
    final_model_latent_save_path = "callbacks/final_model/latent_model.pth"
    results_save_path = "results/train_results.json"

    # Testing Stage
    test_real_img_save_path = "results/test/test_real_images.jpg"
    test_generated_img_path = "results/test/test_predict_images.jpg"
    test_result_save_path = "results/test/test_results.json"
    test_model_encoder_save_path = "callbacks/test_models/tested_encoder.pth"
    test_model_decoder_save_path = "callbacks/test_models/tested_decoder.pth"
    test_model_latent_save_path = "callbacks/test_models/tested_latent.pth"

    # Prediction Stage
    predicted_img_save_path = "prediction_images/generated_images"
    prediction_img_load_path = "prediction_images/real_images"
