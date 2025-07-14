import argparse
import time
import sys


class Config:
    def __init__(self, args=[]):
        ctime = time.ctime()

        parser = argparse.ArgumentParser(
            description="Configuration for the model training and evaluation."
        )

        ## Project and Run
        parser.add_argument("--project", type=str, default="PSOR")
        parser.add_argument("--run_name", type=str, default=f"{ctime}_wandb")
        parser.add_argument("--output_dir", type=str, default=f"{ctime}")

        ## Training parameters
        parser.add_argument("--num_train_epochs", type=int, default=8)
        parser.add_argument("--per_device_train_batch_size", type=int, default=1)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--logging_steps", type=int, default=50)
        parser.add_argument("--eval_steps", type=int, default=2500)
        parser.add_argument("--save_steps", type=int, default=2500)
        parser.add_argument("--wandb_mode", type=str, default="offline")

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
        parser.add_argument("--model_id", type=str, default="assets/Qwen/Qwen2-VL-7B-Instruct")
        
        args = parser.parse_args(args=args)
        for key, value in vars(args).items():
            setattr(self, key, value)

def get_config():
    return Config(sys.argv[1::])


if __name__ == "__main__":
    config = get_config()
    for key, value in vars(config).items():
        print(f"{key}: {value}")
