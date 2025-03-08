import torch
import torch.nn as nn
from src.exception.exception import ExceptionNetwork, sys

class Latent_Space(nn.Module):
    def __init__(self, num_embedding=512):
        super(Latent_Space, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=1)

    def latent_space(self, pre_quantized):
        try:
            # (B, C, H, W) -> (B, H, W, C) olacak şekilde permute et
            data_perm = pre_quantized.permute(0, 2, 3, 1)

            # (batch_size, H*W, C) olarak yeniden şekillendir
            quanted_in = data_perm.reshape(data_perm.shape[0], -1, data_perm.shape[-1])

            # Öklid mesafesini hesapla
            dist = torch.cdist(quanted_in, self.embedding.weight[None, :, :]) # (B, H*W, num_embedding)

            # Minimum mesafeli vektörleri bul
            min_distance_indexes = torch.argmin(dist, dim=-1)  # (B, H*W)

            # Embedding'leri al
            quanted_out = self.embedding(min_distance_indexes)  # (B, H*W, C)

            return quanted_in, quanted_out, min_distance_indexes
        
        except Exception as e:
            raise ExceptionNetwork(e, sys)
    
    def forward(self, pre_quantized):
        try:
            # Vektörleri hesapla
            quanted_in, quanted_out, min_distance_indexes = self.latent_space(pre_quantized)

            # Quantization loss hesapla
            loss1 = torch.mean((quanted_out.detach() - quanted_in) ** 2)  # Encoder’ı cezalandır
            loss2 = torch.mean((quanted_out - quanted_in.detach()) ** 2)  # Embedding’i cezalandır
            loss_quantize = loss1 + (0.25 * loss2)  

            # Gradient akışı için trick
            quanted_out = quanted_in + (quanted_out - quanted_in).detach()

            # Decoder girişi için yeniden şekillendir (B, C, H, W)
            B, C, H, W = pre_quantized.shape
            quanted_out_reshaped = quanted_out.reshape(B, H, W, C).permute(0, 3, 1, 2)  

            return quanted_out_reshaped, loss_quantize
        
        except Exception as e:
            raise ExceptionNetwork(e, sys)
