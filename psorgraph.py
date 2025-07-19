import numpy as np
import json


def enc(lst):
    return ",".join(list(map(str, lst)))


class BBox:
    def __init__(self, bbox: dict):
        self.x1 = bbox["x1"]
        self.y1 = bbox["y1"]
        self.x2 = bbox["x2"]
        self.y2 = bbox["y2"]

    def intersection(self, bbox):
        x1 = max(self.x1, bbox.x1)
        y1 = max(self.y1, bbox.y1)
        x2 = min(self.x2, bbox.x2)
        y2 = min(self.y2, bbox.y2)
        return BBox({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    def area(self):
        w = max(self.x2 - self.x1, 0)
        h = max(self.y2 - self.y1, 0)
        return h * w

    def iou(self, bbox):
        s = self.intersection(bbox).area()
        u = self.area() + bbox.area() - s
        return s / u

    def scale(self, r_x, r_y):
        self.x1 = self.x1 * r_x
        self.y1 = self.y1 * r_y
        self.x2 = self.x2 * r_x
        self.y2 = self.y2 * r_y

    def __str__(self):
        return str({"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2})

class COCOAnno:
    def __init__(self, anno: dict, categories=None):
        x, y, w, h = anno["box"]
        self.bbox = BBox({"x1": x, "y1": y, "x2": x + w, "y2": y + h})
        
        self.category = anno["category_id"] if categories==None else categories[anno["category_id"]]

    def __str__(self):
        return str({"category": self.category, "bbox": str(self.bbox)})

class PSORGraph:
    def __init__(self, data, categories=None):
        table = dict((enc(x["condition"]), x) for x in data["psor_samples"])
        annos = dict((i, COCOAnno(anno, categories)) for i, anno in enumerate(data["annotations"]))

        self.matched_threshold = 0.5

        self.table = table
        self.annos = annos
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
