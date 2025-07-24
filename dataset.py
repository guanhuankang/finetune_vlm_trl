import os
import json
from torch.utils.data import Dataset
from PIL import Image


def format_data(sample):
    system_message = """You are a Vision Language Model specialized in Salient Object Ranking. Detect all salient objects in the user's image and rank them from the most to least salient. Output results in this strict JSON format: {"results": [{"rank": 1,"category": "object_name", "bbox": {"x1": x1:int, "y1": y1:int, "x2": x2:int, "y2": y2:int}}, ..., {"rank": N, "category": "background","bbox": {"x1": 0, "y1": 0, "x2": width, "y2": height}}]}
    Requirements:
    1. Final entry must be background object with its bounding box covering the full image (x1=0, y1=0, x2=width, y2=height).
    2. Bounding boxes use absolute pixel coordinates (x1,y1 = top-left, x2,y2 = bottom-right).
    3. Images typically contain only a few salient objects, with a maximum limit of 10 per image.
    4. Output must be pure JSON with no additional text."""
    # The maximum number of salient objects per image is limited to 10.
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
                    "text": f"This is the input image with height = {sample['input_height']} and width = {sample['input_width']}."
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["label"]}],
        },
    ]

class PSORDataset(Dataset):
    def __init__(self, cfg, split_index:str):
        dataset_path = cfg.dataset_path
        categories_path = cfg.categories_path
        split_start, split_length = tuple(map(int, split_index.split(",")))

        with open(dataset_path, "r") as f:
            dataset = json.load(f)[split_start: split_start + split_length]

        with open(categories_path, "r") as f:
            categories = json.load(f)
            categories = dict((x["id"], x["name"]) for x in categories)

        self.cfg = cfg
        self.categories = categories
        self.dataset = [self.preprocess_psor_sample(x) for x in dataset]

    def __getitem__(self, index):
        sample = self.dataset[index]
        input_width = sample["input_width"]
        input_height = sample["input_height"]
        image = Image.open(
            os.path.join(self.cfg.image_folder_path, sample["name"] + ".jpg")
        ).convert('RGB')
        
        chat_content = format_data({
            "image": image.resize((input_width, input_height)),
            "label": json.dumps({"results": sample["sor"]}),
            "input_width": input_width,
            "input_height": input_height,
        })

        if self.cfg.evaluation:
            sample["chat_content"] = chat_content[0:-1]
            sample["add_generation_prompt"] = True
        else:
            sample["chat_content"] = chat_content
            sample["add_generation_prompt"] = False
        
        sample["image"] = image
        
        return sample

    def __len__(self):
        return len(self.dataset)

    def preprocess_psor_sample(self, raw_sample):
        name = raw_sample["image"]
        height = raw_sample["height"]
        width = raw_sample["width"]
        input_width = self.cfg.input_width
        input_height = self.cfg.input_height

        table = dict(
            (",".join(list(map(str, x["condition"]))), x)
            for x in raw_sample["psor_samples"]
        )

        k = ""
        rank = 1
        annos = raw_sample["annotations"]
        sor = []
        masks = []

        def meet_end():
            return {
                "rank": rank,
                "category": "background",
                "bbox": {"x1": 0, "y1": 0, "x2": input_width, "y2": input_height}
            }

        while True:
            if k == "" and k not in table:
                sor.append(meet_end())
                break

            x = table[k]
            anno_idx = x["groundtruth"][x["optimal_index"]]["anno_idx"]
            if anno_idx == "end":
                sor.append(meet_end())
                break
            else:
                anno_data = annos[anno_idx]
                x1, y1, w, h = anno_data["box"]
                x2, y2 = x1 + w, y1 + h
                sor.append(
                    {
                        "rank": rank,
                        "bbox": {
                            "x1": int(x1 / width * input_width),
                            "y1": int(y1 / height * input_height),
                            "x2": int(x2 / width * input_width),
                            "y2": int(y2 / height * input_height),
                        },
                        "category": self.categories[anno_data["category_id"]],
                    }
                )
                masks.append({"rank": rank, "mask": anno_data["mask"]})
            rank = rank + 1
            k = f"{k},{anno_idx}" if k != "" else f"{anno_idx}"

        sample = {
            "name": name,
            "height": height,
            "width": width,
            "input_height": input_height,
            "input_width": input_width,
            "sor": sor,
            "masks": masks,
            "is_salient": len(sor) > 1, 
        }

        return sample

class EvalImageHandler:
    def __init__(self, cfg):
        self.cfg = cfg
    
    def handle(self, image_path):
        name = os.path.splitext(os.path.basename(image_path))[0]
        input_width = self.cfg.input_width
        input_height = self.cfg.input_height

        image = Image.open(image_path).convert("RGB")
        input_image = image.resize((input_width, input_height))

        width, height = image.size
        return {
            "name": name,
            "width": width,
            "height": height,
            "input_width": input_width,
            "input_height": input_height,
            "image": image,  ## original image
            "chat_content": format_data({
                "image": input_image,
                "input_width": input_width,
                "input_height": input_height,
                "label": ""
            })[0:-1] ## remove assistant
        }

def load_psor_dataset(cfg):
    eval_dataset = PSORDataset(cfg, split_index=cfg.val_split)
    test_dataset = PSORDataset(cfg, split_index=cfg.test_split)
    train_dataset = PSORDataset(cfg, split_index=cfg.train_split)

    return eval_dataset, test_dataset, train_dataset


if __name__ == "__main__":
    from config import get_config

    cfg = get_config()

    eval_dataset, test_dataset, train_dataset = load_psor_dataset(cfg=cfg)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Example of accessing a sample
    # Print the first training sample
    print("Example train sample:", train_dataset[0])
    # print("Example eval sample:", eval_dataset[0])    # Print the first evaluation sample

    handler = EvalImageHandler(cfg=cfg)
    sample = handler.handle(image_path="assets/dataset/images/000000386912.jpg")
    print(sample)