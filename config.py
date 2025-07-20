import argparse
import time
import sys
import secrets
import os

class Config:
    def __init__(self, args=[]):
        # ctime = time.strftime("%Y%m%d_%H%M%S")
        run_id = secrets.token_hex(4)

        parser = argparse.ArgumentParser(
            description="Configuration for the model training and evaluation."
        )

        ## Project and Run
        parser.add_argument("--project", type=str, default="PSOR")
        parser.add_argument("--run_id", type=str, default=run_id)
        parser.add_argument("--output_root", type=str, default="output/")
        
        ## Train or Test Mode
        parser.add_argument('--evaluation', action='store_true', help='Evaluation Mode')

        ## Training parameters
        parser.add_argument("--num_train_epochs", type=int, default=2)
        parser.add_argument("--num_gpus", type=int, default=1)
        parser.add_argument("--per_device_train_batch_size", type=int, default=4)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
        parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
        parser.add_argument("--logging_steps", type=int, default=5)
        parser.add_argument("--eval_steps", type=int, default=625)
        parser.add_argument("--save_steps", type=int, default=625)
        parser.add_argument("--wandb_mode", type=str, default="offline")
        parser.add_argument("--learning_rate", type=float, default=2e-5)

        ## Dataset
        parser.add_argument(
            "--dataset_path", type=str, default="assets/dataset/psor.json"
        )
        parser.add_argument(
            "--categories_path", type=str, default="assets/dataset/categories.json"
        )
        parser.add_argument(
            "--image_folder_path", type=str, default="assets/dataset/images"
        )
        parser.add_argument(
            "--val_test_train_split",
            type=str,
            default="0,200;0,5000;5000,10000",
            help="start,length",
        )

        ## Model parameters
        # parser.add_argument('--sam_checkpoint', type=str, default='assets/sam_vit_h_4b8939.pth')
        parser.add_argument(
            "--model_id", type=str, default="assets/Qwen/Qwen2-VL-7B-Instruct"
        )

        args = parser.parse_args(args=args)
        for key, value in vars(args).items():
            setattr(self, key, value)

def get_config():
    cfg = Config(sys.argv[1::])
    cfg.output_dir = os.path.join(cfg.output_root, cfg.run_id)
    return cfg

if __name__ == "__main__":
    config = get_config()
    for key, value in vars(config).items():
        print(f"{key}: {value}")
