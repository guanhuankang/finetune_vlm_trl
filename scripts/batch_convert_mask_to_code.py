import json
import tqdm
from mask_tokenizer import MaskTokenizer
from pycocotools.mask import decode as coco_mask_decode
from PIL import Image

def batch_convert_mask_to_code(filename, outname, ckp):
    mt = MaskTokenizer(checkpoint_path=ckp)

    with open(filename, "r") as f:
        data = json.load(f)

    mask_to_code = {}

    for item in tqdm.tqdm(data):
        for i, anno in enumerate(item["annotations"]):
            x, y, w, h = tuple(map(int, anno["box"]))
            mask_np = coco_mask_decode(anno["mask"])
            mask_np = mask_np[y:y+max(h,1), x:x+max(w,1)]
            code = mt.encode2code(mask_np=mask_np)
            mask_to_code[f"{item['image']}_{i}"] = code.view(-1).tolist()

    with open(outname, "w") as f:
        json.dump(mask_to_code, f)

    return len(mask_to_code)

filename = "assets/dataset/psor.json"
outname = "assets/dataset/psor_maskcode_bbox.json"
ckp = "assets/1d_tokenizer/tokenizer_titok_l32_imagenet"

n = batch_convert_mask_to_code(
    filename=filename,
    outname=outname,
    ckp=ckp
)

print(n, "done")
