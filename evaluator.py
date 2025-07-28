import os
import json
import numpy as np
from PIL import Image

from utils import BBox

def enc(lst):
    return ",".join(list(map(str, lst)))

class COCOAnno:
    def __init__(self, anno: dict, categories=None):
        x, y, w, h = anno["box"]
        self.bbox = BBox({"x1": x, "y1": y, "x2": x + w, "y2": y + h})

        self.category = (
            anno["category_id"]
            if categories == None
            else categories[anno["category_id"]]
        )

    def __str__(self):
        return str({"category": self.category, "bbox": str(self.bbox)})


class PSORGraph:
    def __init__(self, data, categories=None):
        table = dict((enc(x["condition"]), x) for x in data["psor_samples"])
        annos = dict(
            (i, COCOAnno(anno, categories))
            for i, anno in enumerate(data["annotations"])
        )

        self.table = table
        self.annos = annos
        self.matched_threshold = 0.5
        self.name = data["image"]
        self.post_init()

    def post_init(self):
        if "" not in self.table:
            self.table[""] = {
                "condition": [],
                "parent_weight": 1.0,
                "max_reward": 1.0,
                "optimal_index": 0,
                "groundtruth": [
                    {
                        "anno_idx": "end",
                        "weight": 1.0,
                        "action_reward": 1.0,
                        "max_reward": 1.0,
                        "cumulative_reward": 1.0,
                    }
                ],
            }
        self.max_reward = self.table[""]["max_reward"]

    def match(self, obj, state: list):
        """obj: {"bbox": BBox, "category": str}"""
        node_data = self.table[enc(state)]
        children_data = node_data["groundtruth"]

        iou_scores = []
        for c in children_data:
            anno_idx = c["anno_idx"]
            if anno_idx == "end":
                score = 1.0 if obj["category"] == "background" else 0.0
            else:
                score = self.annos[anno_idx].bbox.iou(obj["bbox"])
            iou_scores.append(score)

        matched_index = np.argmax(iou_scores)

        if iou_scores[matched_index] >= self.matched_threshold:
            return {
                "anno_idx": children_data[matched_index]["anno_idx"],
                "iou": iou_scores[matched_index],
                "action_reward": children_data[matched_index]["action_reward"],
                "max_action_reward": children_data[node_data["optimal_index"]][
                    "action_reward"
                ],
            }
        else:
            return {
                "anno_idx": None,
                "iou": 0.0,
                "action_reward": 0.0,
                "max_action_reward": children_data[node_data["optimal_index"]][
                    "action_reward"
                ],
            }


class Evaluator:
    def __init__(self, config):
        with open(config.dataset_path, "r") as f:
            dataset = json.load(f)

        with open(config.categories_path, "r") as f:
            categories = json.load(f)
            categories = dict((x["id"], x["name"]) for x in categories)

        self.name_map_to_dataset_index = dict(
            (x["image"], i) for i, x in enumerate(dataset)
        )

        self.graphs = [PSORGraph(data=data, categories=categories)
                       for data in dataset]
        self.infos = [
            {
                "name": data["image"],
                "width": data["width"],
                "height": data["height"],
                "image_path": os.path.join(config.image_folder_path, data["image"]+".jpg")
            }
            for data in dataset
        ]

        self.evaluation = config.evaluation

        self.init()

    def init(self):
        self.results = []

    def __len__(self):
        return len(self.results)

    def get_info(self, name):
        index = self.name_map_to_dataset_index[name]
        return self.infos[index]

    def get_image(self, name):
        info = self.get_info(name)
        return Image.open(info["image_path"]).convert('RGB')

    def calc(self, generated_lst, graph):
        state = []
        scores = {
            "top1_advantage": 0.0,
            "top1_advantage_iou": 0.0,
            "advantage": 0.0,
            "advantage_iou": 0.0,
        }
        sum_action_rewards = 0.0
        sum_iou_aware_action_rewards = 0.0

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

        return scores

    def update(
        self,
        name: str,
        width: int,
        height: int,
        input_width: int,
        input_height: int,
        results: dict,
    ):
        """
        "results": list of detected objects
            {"rank": 1,"category": "object_name", "bbox": {"x1": 0, "y1": 0, "x2": 0, "y2": 0}}
        """
        generated_lst = []
        for obj in results["results"]:
            bbox = BBox(obj["bbox"])
            bbox.scale(r_x = width / input_width, r_y = height / input_height)
            generated_lst.append({
                "rank": obj["rank"],
                "category": obj["category"],
                "bbox": bbox
            })

        generated_lst.sort(key=lambda obj: obj["rank"])

        if name in self.name_map_to_dataset_index:
            graph = self.graphs[self.name_map_to_dataset_index[name]]
            self.results.append(self.calc(generated_lst, graph))
            return generated_lst
        else:
            return generated_lst

    def sum(self):
        ret = {}
        for scores in self.results:
            for k, v in scores.items():
                ret[k] = ret.get(k, 0.0) + v
        return ret | {"count": len(self)}

    def average(self):
        n = len(self)
        if n==0:
            return {"count": n}
        
        ret = self.sum()
        for k, v in ret.items():
            ret[k] = v / n
        return ret | {"count": n}
