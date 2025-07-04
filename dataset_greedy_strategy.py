import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
import json
import pycocotools.mask as cocomask

import os
from PIL import Image

class GreedyPSORDataset(Dataset):
    def __init__(self, config, split):
        super().__init__()

        dataset_file = {
            "train": config.dataset_train_file,
            "val": config.dataset_val_file
        }[split]
        
        with open(dataset_file, "r") as f:
            self.data = json.load(f)
        
        self.image_root = config.image_root
        self.split = split
        self.max_objects = config.max_objects

    def __len__(self):
        return len(self.data)
    
    def greedy_strategy(self, psor_samples):
        ''' return a list of samples constructing the best ranking '''
        enc = lambda x: ",".join(list(map(str, x)))
        table = dict( ( enc(x["condition"]), x ) for x in psor_samples)
        
        assert "" in table, ("psor_samples", psor_samples)
        best_seq = []
        k = ""
        while not k.endswith("end"):
            best_seq.append(table[k])
            
            nxt_obj = table[k]["groundtruth"][table[k]["optimal_index"]]["anno_idx"]
            k = enc(table[k]["condition"] + [nxt_obj, ])
        
        return best_seq ## debug here        
    
    def __getitem__(self, index):
        image_data = self.data[index]
        image_name = image_data["image"]
        
        if len(image_data["psor_samples"]) <= 0:
            print(f"{image_name} is empty sample.")
            return self.__getitem__(index=(index+1) % len(self))
        
        annos = image_data["annotations"]
        sample = np.random.choice(self.greedy_strategy(image_data["psor_samples"]))
        
        pil_image = Image.open(os.path.join(self.image_root, image_name+".jpg"))
        image = transforms.PILToTensor()(pil_image)
        
        seq_images = [image]
        seq_meta_data = [{"label": "input"},]
        for idx in sample["condition"]:
            mask = torch.tensor(cocomask.decode(annos[idx]["mask"]))
            mask_image = mask[None, :, :]
            
            seq_images.append(mask_image)
            seq_meta_data.append({"label": annos[idx]["category_id"]})
        
        ## select the optimal
        gt = sample["groundtruth"][sample["optimal_index"]]
        gt_mask = torch.tensor(cocomask.decode(annos[gt["anno_idx"]]["mask"]))
        gt_mask = gt_mask[None, :, :]
        gt_label = annos[gt["anno_idx"]]["category_id"]
        
        return {
            "image_name": image_name, # str
            "height": image_data["height"],  # int
            "width": image_data["width"],  # int
            "image": image, # 3, h, w
        }