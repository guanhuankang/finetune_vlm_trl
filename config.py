import argparse
import sys
import secrets


class Config:
    def __init__(self, args=[]):
        # ctime = time.strftime("%Y%m%d_%H%M%S")
        run_name = secrets.token_hex(4)

        parser = argparse.ArgumentParser(
            description="Configuration for the model training and evaluation."
        )

        # Project and Run
        parser.add_argument("--project", type=str, default="PSOR")
        parser.add_argument("--run_name", type=str, default=run_name)
        parser.add_argument("--output_dir", type=str, default="output/")
        parser.add_argument("--runs_dir", type=str, default="runs/")
        parser.add_argument(
            "--evaluation", action="store_true", help="Evaluation Mode")

        # Model
        parser.add_argument("--model_id", type=str,
                            default="assets/Qwen/Qwen2-VL-7B-Instruct")
        # parser.add_argument('--sam_checkpoint', type=str, default='assets/sam_vit_h_4b8939.pth')

        # Training parameters
        parser.add_argument("--num_train_epochs", type=int, default=2)
        parser.add_argument("--num_gpus", type=int, default=1)
        parser.add_argument("--per_device_train_batch_size",
                            type=int, default=4)
        parser.add_argument("--gradient_accumulation_steps",
                            type=int, default=4)
        parser.add_argument("--per_device_eval_batch_size",
                            type=int, default=1)
        parser.add_argument("--logging_steps", type=int, default=5)
        parser.add_argument("--eval_steps", type=int, default=625)
        parser.add_argument("--quick_eval", action="store_true", help="skip slow eval")
        parser.add_argument("--save_steps", type=int, default=625)
        parser.add_argument("--wandb_mode", type=str, default="offline")
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--input_width", type=int, default=1024)
        parser.add_argument("--input_height", type=int, default=1024)

        # Dataset
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
            "--val_split", type=str, default="0,200", help="start,length"
        )
        parser.add_argument(
            "--test_split", type=str, default="0,5000", help="start,length"
        )
        parser.add_argument(
            "--train_split", type=str, default="5000,10000", help="start,length"
        )

        # Visualization
        parser.add_argument("--n_image_visualization", type=int, default=10)

        args = parser.parse_args(args=args)
        for key, value in vars(args).items():
            setattr(self, key, value)


def get_config(args=[]):
    cfg = Config(args + sys.argv[1::])
    cfg.run_id = f"{cfg.run_name}-{secrets.token_hex(2)}"
    return cfg


if __name__ == "__main__":
    config = get_config()
    for key, value in vars(config).items():
        print(f"{key}: {value}")
