import os
import json
import wandb
from psorgraph import PSORGraph, BBox


class Evaluator:
    def __init__(self, cfg):
        dataset_path = cfg.dataset_path
        categories_path = cfg.categories_path
        image_folder_path = cfg.image_folder_path
        val_split = cfg.val_test_train_split.split(";")[0]

        start, length = tuple(map(int, val_split.split(",")))

        with open(dataset_path, "r") as f:
            dataset = json.load(f)[start : start + length]

        with open(categories_path, "r") as f:
            categories = json.load(f)
            categories = dict((x["id"], x["name"]) for x in categories)

        self.name_map_to_dataset_index = dict(
            (x["image"], i) for i, x in enumerate(dataset)
        )

        self.graphs = [
            PSORGraph(
                data=data,
                categories=categories,
                image_path=os.path.join(image_folder_path, data["image"] + ".jpg"),
            )
            for data in dataset
        ]
        wandb.log({"The Size of Evaluation Graphs": len(self.graphs)})

        self.init()

    def init(self):
        self.results = []

    def update(self, x: dict):
        """x is a dict with keys:
        "name":
        "width":
        "height":
        "input_width":
        "input_height":
        "results": list of detected objects
            {"rank": 1,"category": "object_name", "bbox": {"x1": 0, "y1": 0, "x2": 0, "y2": 0}}
        """
        graph = self.graphs[self.name_map_to_dataset_index[x["name"]]]

        state = []
        scores = {
            "top1_advantage": 0.0,
            "top1_advantage_iou": 0.0,
            "advantage": 0.0,
            "advantage_iou": 0.0,
        }
        sum_action_rewards = 0.0
        sum_iou_aware_action_rewards = 0.0
        ratio_x = x["width"] / x["input_width"]
        ratio_y = x["height"] / x["input_height"]

        generated_lst = []
        for obj in x["results"]:
            bbox = BBox(obj["bbox"])
            bbox.scale(r_x=ratio_x, r_y=ratio_y)
            obj["bbox"] = bbox
            generated_lst.append(obj)

        generated_lst = sorted(x["results"], key=lambda obj: obj["rank"])

        wandb.log(
            {
                "visuals": wandb.Image(
                    graph.visualize(generated_lst), caption=x["name"]
                )
            }
        )

        for obj in generated_lst:
            y = graph.match(obj, state)

            if obj["rank"] == 1:
                scores["top1_advantage"] = y["action_reward"] / y["max_action_reward"]
                scores["top1_advantage_iou"] = y["iou"]

            sum_action_rewards += y["action_reward"]
            sum_iou_aware_action_rewards += y["action_reward"] * y["iou"]

            state.append(y["anno_idx"])
            if y["anno_idx"] == "end" or y["anno_idx"] == None:  # early stop
                break

        scores["advantage"] = sum_action_rewards / graph.max_reward
        scores["advantage_iou"] = sum_iou_aware_action_rewards / (
            sum_action_rewards + 1e-6
        )

        self.results.append(scores)

    def sum(self):
        ret = {}
        for scores in self.results:
            for k, v in scores.items():
                ret[k] = ret.get(k, 0.0) + v
        return ret

    def average(self):
        n = len(self.results)
        ret = self.sum()
        for k, v in ret.items():
            ret[k] = v / n
        return ret
