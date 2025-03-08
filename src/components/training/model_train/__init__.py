import tqdm
from src.exception.exception import ExceptionNetwork,sys


def model_training(train_dataloader, optimizer, loss_fn, encoder, decoder, latent_space ,device):
    try: 
        encoder.train()
        decoder.train()
        latent_space.train()
        
        train_loss_value = 0

        progres_bar=tqdm.tqdm(range(len(train_dataloader)),"Training progress")
        
        for batch,img in enumerate(train_dataloader):
            
            img=img.to(device)
            
            optimizer.zero_grad()
            out_encoder=encoder(img)
            out_latent,loss_quantize = latent_space(out_encoder)
            out_decoder=decoder(out_latent)
            
            reconstruction_loss=loss_fn(out_decoder,img)
            loss = reconstruction_loss+loss_quantize
            
            loss.backward()
            optimizer.step()
            progres_bar.update(1)
            
            train_loss_value+=loss.item()
        
        total_loss = train_loss_value/(batch+1)
        
        progres_bar.set_postfix({"train_loss":total_loss})
                
        return total_loss
   
    except Exception as e:
            raise ExceptionNetwork(e,sys)
    

        