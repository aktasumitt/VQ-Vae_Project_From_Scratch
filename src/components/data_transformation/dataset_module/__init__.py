from torch.utils.data import Dataset
from PIL import Image


class DatasetModule(Dataset):
    def __init__(self,full_img_list,transformer):
        super(DatasetModule,self).__init__()
        self.transformer=transformer
        self.full_img_list=full_img_list
        
    def __len__(self):
        return len(self.full_img_list)

    def __getitem__(self, index):
        
        img_path=self.full_img_list[index]
        image=Image.open(img_path).convert("RGB")
        image=self.transformer(image)
                
        return image