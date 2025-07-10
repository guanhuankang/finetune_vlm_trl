import os
import json
from torch.utils.data import Dataset
from PIL import Image


def format_data(sample):
    system_message = """You are a Vision Language Model specialized in Salient Object Ranking, a task focused on predicting the sequential order of human attention shifts among objects in a scene. Your task is to detect salient objects, and rank them from most to least salient, and output the results in the following strict JSON format:
    ```json
    {
        "results": [
            {"rank": 1, "category": "dog", "bbox": {"x1": 50, "y1": 120, "x2": 200, "y2": 300}},
            {"rank": 2, "category": "car", "bbox": {"x1": 250, "y1": 80, "x2": 400, "y2": 180}},
            {"rank": 3, "category": "tree", "bbox": {"x1": 10, "y1": 50, "x2": 150, "y2": 350}}
        ]
    }
    ```
    """
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
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["label"]}],
        },
    ]


class PSORDataset(Dataset):
    def __init__(self, dataset_path, image_folder_path, categories_path, split, input_resolution=(1024, 1024)):
        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        with open(categories_path, "r") as f:
            categories = json.load(f)
            categories = dict((x["id"], x["name"]) for x in categories)

        # Splitting
        if split == "val" or split == "test":
            dataset = dataset[0:5]
        elif split == "train":
            dataset = dataset[5::]
        else:
            dataset = dataset

        self.categories = categories
        self.image_folder_path = image_folder_path
        self.input_resolution = input_resolution
        self.dataset = [self.preprocess_psor_sample(x) for x in dataset]

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample["image"] = Image.open(os.path.join(
            self.image_folder_path, sample["name"]+".jpg")).resize(self.input_resolution)
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
        while True:
            x = table[k]
            anno_idx = x["groundtruth"][x["optimal_index"]]["anno_idx"]
            if anno_idx == "end":
                break
            else:
                anno_data = annos[anno_idx]
                x1, y1, w, h = anno_data["box"]
                x2, y2 = x1 + w, y1 + h
                sor.append({
                    "rank": rank,
                    "bbox": {
                        "x1": int(x1/width*input_width),
                        "y1": int(y1/height*input_height),
                        "x2": int(x2/width*input_width),
                        "y2": int(y2/height*input_height)
                    },
                    "category": self.categories[anno_data["category_id"]]
                })
                masks.append({
                    "rank": rank,
                    "mask": anno_data["mask"]
                })

                rank = rank + 1
                k = f'{k},{anno_idx}' if k != "" else f'{anno_idx}'

        sample = {
            "name": name,
            "height": height,
            "width": width,
            "input_height": input_height,
            "input_width": input_width,
            "sor": sor,
            "masks": masks
        }

        return sample


def load_psor_dataset():
    dataset_path = "minidataset/psor_examples.json"
    categories_path = "minidataset/categories.json"
    image_folder_path = "minidataset/examples"

    train_dataset = PSORDataset(dataset_path=dataset_path, image_folder_path=image_folder_path,
                                categories_path=categories_path, split="train")
    eval_dataset = PSORDataset(dataset_path=dataset_path, image_folder_path=image_folder_path,
                                categories_path=categories_path, split="val")
    test_dataset = PSORDataset(dataset_path=dataset_path, image_folder_path=image_folder_path,
                                categories_path=categories_path, split="test")

    return train_dataset, eval_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, eval_dataset, test_dataset = load_psor_dataset()

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Example of accessing a sample
    # Print the first training sample
    print("Example train sample:", train_dataset[0])
    # print("Example eval sample:", eval_dataset[0])    # Print the first evaluation sample
