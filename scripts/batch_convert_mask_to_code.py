import json
import tqdm
from mask_tokenizer import MaskTokenizer

def batch_convert_mask_to_code(filename, outname, ckp):
    mt = MaskTokenizer(checkpoint_path=ckp)

    with open(filename, "r") as f:
        data = json.load(f)

    mask_to_code = {}

    for x in tqdm.tqdm(data):
        for i, anno in enumerate(x["annotations"]):
            code = mt.encode2code(mask_rle=anno["mask"])
            code = code.view(-1).tolist()
            mask_to_code[f"{x['image']}_{i}"] = code

    with open(outname, "w") as f:
        json.dump(mask_to_code, f)

    return len(mask_to_code)

filename = "/home/huankguan2/projects/psor2025/finetune_vlm_trl/assets/dataset/psor.json"
outname = "/home/huankguan2/projects/psor2025/finetune_vlm_trl/assets/dataset/psor_maskcode.json"
ckp = "/home/huankguan2/projects/psor2025/finetune_vlm_trl/assets/1d_tokenizer/tokenizer_titok_l32_imagenet"

n = batch_convert_mask_to_code(
    filename=filename,
    outname=outname,
    ckp=ckp
)

print(n, "done")