import os
import json
import numpy as np
import datetime
from pycocotools.mask import decode
import tqdm

from evaluator import Evaluator, BBox
from config import PSORConfig

class Anno2Metrics:
    def __init__(self, run_name, filename):
        self.config = PSORConfig(
            project="anno2metrics", 
            run_name=run_name, 
            n_image_visualization=4, 
            evaluation=True
        )
        self.evaluator = Evaluator(self.config)
        self.data = self.load(filename)

        self.output_dir = os.path.join(self.config.output_dir, self.config.run_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.config.save_pretrained(self.output_dir)

    def dump(self, data, filename):
        with open(os.path.join(self.output_dir, filename), "w") as f:
            json.dump(data, f)

    def load(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)

        def getbox(obj):
            x, y, w, h = obj["box"]
            return BBox({"x1": x, "y1": y, "x2": x+w, "y2": y+h})

        def getbackground(r):
            return [{
                "rank": r,
                "category": "background",
                "bbox": BBox({"x1": 0, "y1": 0, "x2": 0, "y2": 0}),
                "mask": np.zeros((1,1)),
            }]

        lst = []
        for x in tqdm.tqdm(data):
            lst.append({
                "name": x["image"],
                "width": x["width"],
                "height": x["height"],
                "input_width": x["width"],
                "input_height": x["height"],
                "results": [
                    {
                        "rank": int(i+1),
                        "category": "unknown",
                        "bbox": getbox(obj),
                        "mask": decode(obj["mask"])
                    }
                    for i, obj in enumerate(x["annotations"])
                ] + getbackground(len(x["annotations"])+1)
            })

        return lst
    
    def print_results(self):
        self.evaluator.init()
        for x in self.data:
            self.evaluator.update(**x)
        metrics = self.evaluator.average()
        metrics["run_name"] = self.config.run_name
        metrics["split"] = self.config.run_name.split("_")[-1]

        print(self.config.run_name, metrics)
        self.dump(metrics, "metrics.json")
        return metrics

if __name__=="__main__":
    import pandas as pds

    path = "assets/compared_methods"
    
    records = []

    for x in os.listdir(path):
        records.append(Anno2Metrics(
            run_name=x.split(".")[0],
            filename=os.path.join(path, x)
        ).print_results())
    
    df = pds.DataFrame.from_dict(records)
    df.to_excel("comparison.xlsx")
    print(df)