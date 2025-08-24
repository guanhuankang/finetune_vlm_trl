import torch
from PIL import Image
import numpy as np
from bytedance_1d_tokenizer.titok import TiTok
from pycocotools.mask import decode as coco_mask_decode

class MaskTokenizer:
    def __init__(self, checkpoint_path):
        self.tokenizer = TiTok.from_pretrained(checkpoint_path).to("cuda")
        self.tokenizer.eval()
        self.tokenizer.requires_grad_(False)

    def encode2code(self, mask_rle):
        mask = coco_mask_decode(mask_rle)
        mask_3c = np.repeat(
            mask[..., None], 3, axis=-1
        )  # or np.stack([mask]*3, axis=-1)
        input_image = Image.fromarray((mask_3c * 255).astype(np.uint8)).resize(
            (256, 256)
        )
        input_tensor = (
            torch.from_numpy(np.array(input_image))
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            / 255.0
        )

        code = self.tokenizer.encode(input_tensor.to("cuda"))[1]["min_encoding_indices"]
        return code

    def code2mask(self, code, size=(256, 256)):
        output_tensor = self.tokenizer.decode_tokens(code)
        output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
        output_tensor = torch.nn.functional.interpolate(output_tensor, size=size, mode="bilinear")
        output = (
            (output_tensor * 255.0)
            .permute(0, 2, 3, 1)
            .to("cpu", dtype=torch.uint8)
            .numpy()[0]
        )
        return output[:, :, 0]

if __name__ == "__main__":
    mt = MaskTokenizer(checkpoint_path="assets/mask_tokenizer/tokenizer_titok_l32_imagenet")
    mask_rle = {
        "size": [480, 640],
        "counts": "]UZ34k>2Ld0]O2N0PBUOP>f072N40L3M4L=_OVVl5",
    }
    code = mt.encode2code(mask_rle)
    output = mt.code2mask(code, size=mask_rle["size"])

    print(code.view(-1).tolist())
    Image.fromarray(coco_mask_decode(mask_rle) * 255).save("output/mt_input.jpg")
    Image.fromarray(output).save("output/mt_output.jpg")