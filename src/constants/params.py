class Params:

    # Data Transformation Stage
    normalize_mean = (0.5,)
    normalize_std = (0.5,)
    valid_rate= 0.2
    test_rate= 0.1

    # Model Ingestion Stage
    channel_size = 3
    hidden_dim = 256
    num_embedding = 512
    

    # Training Stage
    batch_size = 50
    noise_dim = 100
    device = "cuda"
    learning_rate = 2e-4
    beta1 = 0.9
    beta2 = 0.999
    epochs = 100
    load_checkpoint=False


    # Test Step
    load_checkpoints_for_test=False
    save_tested_model =False
    
    # Prediction step
    img_size=128

    
    
    