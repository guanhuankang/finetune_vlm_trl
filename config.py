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
        run_name: str = generate_run_name(),
        output_dir: str = "output",
        adapter_path: str = "",
        evaluation: bool = False,
        base_model_id: str = "assets/Qwen/Qwen2.5-VL-3B-Instruct",
        num_train_epochs: int = 2,
        num_gpus: int = 1,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        per_device_eval_batch_size: int = 1,
        logging_steps: int = 5,
        eval_steps: int = 625,
        quick_eval: bool = False,
        save_steps: int = 625,
        wandb_mode: str = "offline",
        learning_rate: float = 2e-5,
        input_width: int = 1036,
        input_height: int = 1036,
        dataset_path: str = "assets/dataset/psor.json",
        categories_path: str = "assets/dataset/categories.json",
        image_folder_path: str = "assets/dataset/images",
        val_split: str = "0,100",
        test_split: str = "0,5000",
        train_split: str = "5000,10000",
        n_image_visualization: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.project = project
        self.run_name = run_name
        self.run_id = generate_run_id(run_name=run_name)
        self.output_dir = output_dir
        self.sft_output_dir = os.path.join(output_dir, run_name)
        self.adapter_path = adapter_path
        self.evaluation = evaluation
        self.base_model_id = base_model_id
        self.num_train_epochs = num_train_epochs
        self.num_gpus = num_gpus
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.quick_eval = quick_eval
        self.save_steps = save_steps
        self.wandb_mode = wandb_mode
        self.learning_rate = learning_rate
        self.input_width = input_width
        self.input_height = input_height
        self.dataset_path = dataset_path
        self.categories_path = categories_path
        self.image_folder_path = image_folder_path
        self.val_split = val_split
        self.test_split = test_split
        self.train_split = train_split
        self.n_image_visualization = n_image_visualization

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
    config = PSORConfig.from_args_and_file()
    print(config)
