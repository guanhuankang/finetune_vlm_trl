from transformers import PretrainedConfig
import argparse
import inspect
import secrets
import time
import os
import json

MODEL_TYPE = "PSOR"


def generate_run_name():
    ctime = time.strftime("%Y%m%d-%H%M%S")
    token = secrets.token_hex(2)
    return f"{ctime}-{token}"


def generate_run_id(run_name: str):
    token = secrets.token_hex(2)
    return f"{run_name}{token}"


class PSORConfig(PretrainedConfig):
    model_type = MODEL_TYPE

    def __init__(
        self,
        project: str = "PSOR",
        run_name: str = "",

        output_dir: str = "output",
        base_model_id: str = "assets/Qwen/Qwen2.5-VL-3B-Instruct",
        dataset_path: str = "assets/dataset/psor.json",
        categories_path: str = "assets/dataset/categories.json",
        image_folder_path: str = "assets/dataset/images",
        sam_checkpoint: str = "assets/sam_vit_h_4b8939.pth",

        val_split: str = "0,100",
        test_split: str = "0,5000",
        train_split: str = "5000,10000",
        num_train_epochs: int = 2,
        num_gpus: int = 1,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        per_device_eval_batch_size: int = 1,
        learning_rate: float = 2e-5,
        input_width: int = 1036,
        input_height: int = 1036,
        logging_steps: int = 5,
        eval_steps: int = 625,
        save_steps: int = 625,
        wandb_mode: str = "offline",
        n_image_visualization: int = 10,

        ckp: int = -1,
        evaluation: bool = False,
        pretrained_path: str = "",
        **kwargs,
    ):
        init_args = [(k, v) for k, v in locals().items() if k not in ('self', 'kwargs', '__class__')]
        init_args.sort()

        super().__init__(**kwargs)

        for k, v in init_args:
            setattr(self, k, v)

        self.run_name = run_name or generate_run_name()
        self.run_id = generate_run_id(self.run_name)
        self.sft_output_dir = os.path.join(output_dir, self.run_name)
        
        if "-3B-" in base_model_id:
            self.mask_decoder_proj2_dim = 2048
        elif "-7B-" in base_model_id:
            self.mask_decoder_proj2_dim = 3584

    @classmethod
    def from_args_and_file(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config_file", type=str, required=False)

        sig = inspect.signature(cls.__init__)
        param_types = {}
        for name, param in sig.parameters.items():
            if name in ("self", "kwargs"):
                continue
            if param.default is not inspect.Parameter.empty:
                param_type = type(param.default)
                param_types[name] = param_type
                if param_type is bool:
                    parser.add_argument(
                        f"--{name}",
                        action="store_true" if not param.default else "store_false",
                    )
                else:
                    parser.add_argument(f"--{name}", type=str)

        args = parser.parse_args()

        if args.pretrained_path:
            args.config_file = args.config_file or os.path.join(args.pretrained_path, "config.json")

        if args.config_file:
            with open(args.config_file, "r") as f:
                config_data = json.load(f)
        else:
            config_data = {}

        for name, val in vars(args).items():
            if name == "config_file" or val is None:
                continue
            if name in param_types:
                expected_type = param_types[name]
                if expected_type is bool:
                    config_data[name] = val
                else:
                    try:
                        config_data[name] = expected_type(val)
                    except Exception as e:
                        raise ValueError(
                            f"Invalid type for argument '{name}': cannot convert '{val}' to {expected_type.__name__}"
                        ) from e
            else:
                config_data[name] = val

        return cls(**config_data)

if __name__ == "__main__":
    config = PSORConfig()
    print(config)
