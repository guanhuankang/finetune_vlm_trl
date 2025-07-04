import argparse

class Config:
    def __init__(self, args=[]):
        parser = argparse.ArgumentParser(description="Configuration for the model training and evaluation.")

        ## Model parameters
        # parser.add_argument('--sam_checkpoint', type=str, default='assets/sam_vit_h_4b8939.pth')
        
        # parser.add_argument('--clip_vision_model', type=str, default='openai/clip-vit-large-patch14')
        # parser.add_argument('--clip_vision_model_image_size', type=int, default='224')
        
        parser.add_argument('--tower_image_size', type=int, default=448)
        
        parser.add_argument('--transformer_width', type=int, default=768)
        parser.add_argument('--transformer_layers', type=int, default=12)
        parser.add_argument('--transformer_heads', type=int, default=8)
        
        parser.add_argument('--vision_tower', type=str, default='PE-Spatial-G14-448')
        parser.add_argument('--vision_tower_width', type=int, default=1536)
        parser.add_argument('--vision_tower_tokens', type=int, default=256)
        
        parser.add_argument('--trajectory_tower', type=str, default='PE-Lang-L14-448')
        parser.add_argument('--trajectory_tower_width', type=int, default=1024)
        parser.add_argument('--trajectory_tower_tokens', type=int, default=16)
        
        parser.add_argument('--max_objects', type=int, default=10)
        parser.add_argument('--num_categories', type=int, default=92, help='COCO Category')
        parser.add_argument('--predition_terms', type=str, default="mask,label,prob")
        
        ## Training parameters
        parser.add_argument('--dataset_train_file', type=str, default='assets/dataset/psor_examples.json')
        parser.add_argument('--dataset_val_file', type=str, default='assets/dataset/psor_test23.json')
        parser.add_argument('--image_root', type=str, default='assets/dataset/images')
        
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--max_epochs', type=int, default=10)
        parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
        parser.add_argument('--K_points', type=int, default=1e9)
        
        parser.add_argument('--ce_loss_weight', type=float, default=1.0)
        parser.add_argument('--dice_loss_weight', type=float, default=1.0)
        parser.add_argument('--mask_weight', type=float, default=1.0)
        parser.add_argument('--prob_weight', type=float, default=1.0)
        parser.add_argument('--label_weight', type=float, default=1.0)
        
        
        args = parser.parse_args(args=args)
        for key, value in vars(args).items():
            setattr(self, key, value)

if __name__ == "__main__":
    import sys
    config = Config(sys.argv[1::])
    for key, value in vars(config).items():
        print(f"{key}: {value}")