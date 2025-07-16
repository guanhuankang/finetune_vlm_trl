import os
import json
from torch.utils.data import Dataset
from PIL import Image


def format_data(sample):
    system_message = """You are a Vision Language Model specialized in Salient Object Ranking. Detect all salient objects in the user's image and rank them from the most to least salient. Output results in this strict JSON format: {"results": [{"rank": 1,"category": "object_name", "bbox": {"x1": 0, "y1": 0, "x2": 0, "y2": 0}}, ..., {"rank": N, "category": "background","message": "No additional salient objects detected."}]}
    Requirements:
    1. Final entry must be background object with the specified message;
    2. Bounding boxes use absolute pixel coordinates (x1,y1 = top-left, x2,y2 = bottom-right);
    3. Output must be pure JSON with no additional text."""
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
                    "text": "This is the input image."
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["label"]}],
        },
    ]


class PSORDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        image_folder_path,
        categories_path,
        split_start,
        split_length,
        input_resolution=(1024, 1024),
    ):
        with open(dataset_path, "r") as f:
            dataset = json.load(f)[split_start : split_start + split_length]

        with open(categories_path, "r") as f:
            categories = json.load(f)
            categories = dict((x["id"], x["name"]) for x in categories)

        self.categories = categories
        self.image_folder_path = image_folder_path
        self.input_resolution = input_resolution
        self.dataset = [self.preprocess_psor_sample(x) for x in dataset]

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample["image"] = Image.open(
            os.path.join(self.image_folder_path, sample["name"] + ".jpg")
        ).resize(self.input_resolution)
        sample["label"] = json.dumps({"results": sample["sor"]})
        return format_data(sample)

    def __len__(self):
        return len(self.dataset)

    def preprocess_psor_sample(self, raw_sample):
        name = raw_sample["image"]
        height = raw_sample["height"]
        width = raw_sample["width"]
        input_height = self.input_resolution[0]
        input_width = self.input_resolution[1]

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
            sor.append(
                {
                    "rank": rank,
                    "category": "background",
                    "msg": "No additional salient objects detected.",
                }
            )

        while True:
            if k == "" and k not in table:
                meet_end()
                break

            x = table[k]
            anno_idx = x["groundtruth"][x["optimal_index"]]["anno_idx"]
            if anno_idx == "end":
                meet_end()
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
        }

        return sample


def load_psor_dataset(cfg):
    dataset_path = cfg.dataset_path
    categories_path = cfg.categories_path
    image_folder_path = cfg.image_folder_path
    split_indexs = cfg.val_test_train_split

    si = tuple(map(int, split_indexs.replace(";", ",").split(",")))

    eval_dataset = PSORDataset(
        dataset_path=dataset_path,
        image_folder_path=image_folder_path,
        categories_path=categories_path,
        split_start=si[0],
        split_length=si[1],
    )
    test_dataset = PSORDataset(
        dataset_path=dataset_path,
        image_folder_path=image_folder_path,
        categories_path=categories_path,
        split_start=si[2],
        split_length=si[3],
    )
    train_dataset = PSORDataset(
        dataset_path=dataset_path,
        image_folder_path=image_folder_path,
        categories_path=categories_path,
        split_start=si[4],
        split_length=si[5],
    )

    return train_dataset, eval_dataset, test_dataset


if __name__ == "__main__":
    from config import get_config

    cfg = get_config()

    train_dataset, eval_dataset, test_dataset = load_psor_dataset(cfg=cfg)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Example of accessing a sample
    # Print the first training sample
    print("Example train sample:", train_dataset[0])
    # print("Example eval sample:", eval_dataset[0])    # Print the first evaluation sample
