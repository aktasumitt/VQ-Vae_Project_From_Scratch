from torchvision.utils import make_grid
import torch
import tqdm
from src.exception.exception import ExceptionNetwork,sys
from torchvision.utils import save_image,Image
# Create Pil Image from tensor batch
def create_pil_image(img):
    try:
                
        img_grid=make_grid(img,10,normalize=True,scale_each=True)
        img_grid_denormalized=(img_grid*255).type(torch.uint8)
                
        # Grid görüntüsünü NumPy formatına çevir
        grid_np = img_grid_denormalized.cpu().detach().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)

        # NumPy array’i PIL Image'e çevir
        grid_pil = Image.fromarray(grid_np)
                
        return grid_pil
                        
    except Exception as e:
        raise ExceptionNetwork(e,sys)
    
    

# Validation Module
def model_validation(valid_dataloader, loss_fn, encoder, decoder, latent_space ,device):
    try: 
        encoder.eval()
        decoder.eval()
        latent_space.eval()
        
        valid_loss_value = 0

        progres_bar=tqdm.tqdm(range(len(valid_dataloader)),"Validation progress")
        
        with torch.no_grad():
            for batch,img in enumerate(valid_dataloader):
                
                img=img.to(device)
                
                out_encoder=encoder(img)
                out_latent,loss_quantize = latent_space(out_encoder)
                out_decoder=decoder(out_latent)
                
                reconstruction_loss=loss_fn(out_decoder,img)
                loss = reconstruction_loss+loss_quantize
                
                valid_loss_value+=loss.item()
                progres_bar.update(1)
                
            total_loss = valid_loss_value/(batch+1)
            progres_bar.set_postfix({"train_loss":total_loss})
            
            real_img=create_pil_image(img)
            predict_img= create_pil_image(out_decoder)
            
            return total_loss, real_img, predict_img
        
    except Exception as e:
        raise ExceptionNetwork(e,sys)

        
        