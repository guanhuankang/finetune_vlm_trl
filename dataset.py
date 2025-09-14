import os
import json
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from pycocotools.mask import decode as coco_mask_decode


def format_data(sample, n_mask_tokens=2):
    mask_placeholder = f"<|mask_start|>{''.join(f'<|mask_{i}|>' for i in range(n_mask_tokens))}<|mask_end|>"

    system_message = """You are a Vision-Language Model (VLM) specialized in Salient Object Ranking (SOR)—the task of modeling how human visual attention dynamically shifts among objects within a scene. Given an input image from the user, your goal is to: 
    (1) Detect visually salient objects - those most likely to capture human attention first.
    (2) Rank them from most to least salient, where Rank 1 = the object that attracts attention first, Rank 2 = next, and so on.
    (3) Provide a bounding box and mask tokens for each ranked object.
    Output Format:
    ```START
    <|object_ref_start|>rank,category,<|box_start|>x1,y1,x2,y2<|box_end|>,{mask_placeholder}<|object_ref_end|>
    <|object_ref_start|>rank,category,<|box_start|>x1,y1,x2,y2<|box_end|>,{mask_placeholder}<|object_ref_end|>
    ...
    ```END
    Guidelines:
    (1) Most images will contain only a few salient objects - limit output to at most 10.
    (2) All bounding boxes must be in absolute pixel coordinates, where:
        a. (x1:int, y1:int) = top-left corner
        b. (x2:int, y2:int) = bottom-right corner
    """.format(mask_placeholder=mask_placeholder)

    text_label = ""
    for obj in sample["label"]:
        text_label += "<|object_ref_start|>{rank},{category},<|box_start|>{x1},{y1},{x2},{y2}<|box_end|>,{mask_placeholder}<|object_ref_end|>".format(
            rank=obj["rank"],
            category=obj["category"],
            x1=obj["bbox"]["x1"],
            y1=obj["bbox"]["y1"],
            x2=obj["bbox"]["x2"],
            y2=obj["bbox"]["y2"],
            mask_placeholder=mask_placeholder
        )
        text_label += "\n"
    text_label = "<|prediction_start|>" + text_label[0:-1] + "<|prediction_end|>"

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": f"This is the input image with height = {sample['input_height']} and width = {sample['input_width']}.",
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": text_label}],
        },
    ]


class PSORDataset(Dataset):
    def __init__(self, config, split_index: str, split):
        dataset_path = config.dataset_path
        categories_path = config.categories_path
        split_start, split_length = tuple(map(int, split_index.split(",")))

        with open(dataset_path, "r") as f:
            dataset = json.load(f)[split_start : split_start + split_length]

        with open(categories_path, "r") as f:
            categories = json.load(f)
            categories = dict((x["id"], x["name"]) for x in categories)

        self.config = config
        self.categories = categories
        self.split = split
        self.dataset = [self.preprocess_psor_sample(x) for x in dataset]

    def __getitem__(self, index):
        sample = self.dataset[index]
        input_width = sample["input_width"]
        input_height = sample["input_height"]
        image = Image.open(
            os.path.join(self.config.image_folder_path, sample["name"] + ".jpg")
        ).convert("RGB")

        chat_content = format_data(
            {
                "image": image.resize((input_width, input_height)),
                "label": sample["sor"],
                "input_width": input_width,
                "input_height": input_height,
            },
            n_mask_tokens=self.config.n_mask_tokens
        )

        if self.split != "train":
            sample["chat_content"] = chat_content[0:-1]
            sample["add_generation_prompt"] = True
        else:
            sample["chat_content"] = chat_content
            sample["add_generation_prompt"] = False

        sample["image"] = image
        sample["masks"] = [x["mask"] for x in sample["sor"]]

        return sample

    def __len__(self):
        return len(self.dataset)

    def preprocess_psor_sample(self, raw_sample):
        name = raw_sample["image"]
        height = raw_sample["height"]
        width = raw_sample["width"]
        input_width = self.config.input_width
        input_height = self.config.input_height

        table = dict(
            (",".join(list(map(str, x["condition"]))), x)
            for x in raw_sample["psor_samples"]
        )

        k = ""
        rank = 1
        annos = raw_sample["annotations"]
        sor = []

        def meet_end():
            return {
                "rank": rank,
                "category": "background",
                "bbox": {"x1": 0, "y1": 0, "x2": input_width, "y2": input_height},
                "mask": np.ones((input_height, input_width)),
            }

        while True:
            if k == "" and k not in table:
                # sor.append(meet_end())
                break

            x = table[k]
            anno_idx = x["groundtruth"][x["optimal_index"]]["anno_idx"]
            if anno_idx == "end":
                # sor.append(meet_end())
                break
            else:
                anno_data = annos[anno_idx]
                x1, y1, w, h = anno_data["box"]
                x2, y2 = x1 + w, y1 + h
                sor.append(
                    {
                        "rank": rank,
                        "category": self.categories[anno_data["category_id"]],
                        "bbox": {
                            "x1": int(x1 / width * input_width),
                            "y1": int(y1 / height * input_height),
                            "x2": int(x2 / width * input_width),
                            "y2": int(y2 / height * input_height),
                        },
                        "mask": coco_mask_decode(anno_data["mask"])
                    }
                )
            rank = rank + 1
            k = f"{k},{anno_idx}" if k != "" else f"{anno_idx}"

        sample = {
            "name": name,
            "height": height,
            "width": width,
            "input_height": input_height,
            "input_width": input_width,
            "sor": sor
        }

        return sample


class EvalImageHandler:
    def __init__(self, config):
        self.config = config

    def handle(self, image_path):
        name = os.path.splitext(os.path.basename(image_path))[0]
        input_width = self.config.input_width
        input_height = self.config.input_height

        image = Image.open(image_path).convert("RGB")
        input_image = image.resize((input_width, input_height))

        width, height = image.size
        return {
            "name": name,
            "width": width,
            "height": height,
            "input_width": input_width,
            "input_height": input_height,
            "image": image,  # original image
            "add_generation_prompt": True,
            "chat_content": format_data(
                {
                    "image": input_image,
                    "input_width": input_width,
                    "input_height": input_height,
                    "label": "",
                }
            )[
                0:-1
            ],  # remove assistant
        }


def load_psor_dataset(config):
    eval_dataset = PSORDataset(config, split_index=config.val_split, split="val")
    test_dataset = PSORDataset(config, split_index=config.test_split, split="test")
    train_dataset = PSORDataset(config, split_index=config.train_split, split="train")

    return eval_dataset, test_dataset, train_dataset


if __name__ == "__main__":
    from config import PSORConfig

    config = PSORConfig.from_args_and_file()

    eval_dataset, test_dataset, train_dataset = load_psor_dataset(config=config)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Example of accessing a sample
    # Print the first training sample
    print("Example train sample:", train_dataset[0])
    # print("Example eval sample:", eval_dataset[0])    # Print the first evaluation sample

    handler = EvalImageHandler(config=config)
    sample = handler.handle(image_path="assets/dataset/images/000000386912.jpg")
    print(sample)
