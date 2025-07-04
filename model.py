import math
import numpy as np
import scipy

import scipy.linalg
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F

# from segment_anything import build_sam_vit_h
# from utils.clip_vision_encoder import ClipVisionTower
from utils.transformer import Transformer
from utils.perception_encoder import get_perception_encoder
from utils.common import PositionalEmbedding2D, MLP, LayerNorm2d
from utils.loss import batch_mask_loss, batch_mask_loss_in_points


class Tower(nn.Module):
    def __init__(
        self, tower_name: str, tower_width: int, d_model: int, channel_side_attn: int
    ):
        super().__init__()

        self.tower = get_perception_encoder(tower_name)

        self.add_pe = PositionalEmbedding2D(d_model=d_model)

        self.projector = nn.Sequential(
            nn.Linear(tower_width, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

        self.channel_side_attention = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, channel_side_attn),
        )

        self.channel_side_projector = MLP(
            in_dim=d_model, mlp_dim=d_model * 4, out_dim=d_model
        )

        self.tower.eval()
        for param in self.tower.parameters():
            param.requires_grad_(False)

    def forward(self, input_images):
        image_embedding = self.tower.forward_features(input_images, layer_idx=-1)
        image_embedding = self.projector(image_embedding)  # B, L, C
        image_embedding = self.add_pe(image_embedding)

        attn = self.channel_side_attention(image_embedding).transpose(
            -1, -2
        )  ## B, n, L
        image_tokens = torch.softmax(attn, dim=-1) @ image_embedding
        image_tokens = image_tokens + self.channel_side_projector(
            image_tokens
        )  ## B, n, C

        return image_tokens, image_embedding


class Vocab(nn.Module):
    def __init__(self, config):
        super().__init__()

        d_model = config.transformer_width

        nt_ans = config.predition_terms.split(",")

        self.vocab = nn.ParameterDict(
            {
                "im_st": nn.Parameter(torch.randn(1, 1, d_model)),
                "im_end": nn.Parameter(torch.randn(1, 1, d_model)),
                "cond_st": nn.Parameter(torch.randn(1, 1, d_model)),
                "cond_end": nn.Parameter(torch.randn(1, 1, d_model)),
                "answer_st": nn.Parameter(torch.randn(1, 1, d_model)),
                "answer_end": nn.Parameter(torch.randn(1, 1, d_model)),
            }
        )

        self.labels = nn.Embedding(config.num_categories, d_model)

        self.answers = nn.Embedding(config.max_objects, d_model * len(nt_ans))

        self.d_model = d_model

    def forward(self, key, indices=None):
        if key == "label":
            return self.labels(indices)[None, :, :]
        elif key == "answer":
            return self.answers(indices).reshape(1, -1, self.d_model)
        else:
            return self.vocab[key]


class MaskDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        width = config.transformer_width
        nt_ans = config.predition_terms.split(",")

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(width, width, kernel_size=4, stride=2, padding=1),
            LayerNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, width, 1),
            LayerNorm2d(width),
            nn.ReLU(),
            nn.ConvTranspose2d(width, width, kernel_size=4, stride=2, padding=1),
            LayerNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, width, 1),
        )

        self.heads = nn.ParameterDict(
            {
                (
                    k,
                    nn.Linear(
                        width,
                        {"mask": width, "label": config.num_categories, "prob": 1}.get(
                            k, 1
                        ),
                    ),
                )
                for k in nt_ans
            }
        )

        self.nt_ans = nt_ans

    def forward(self, prompt, answer_mask, vision_embedding, ori_size):
        B, HW, C = vision_embedding.shape
        H = int(math.sqrt(HW))
        W = int(HW / H)

        vision = self.deconv(vision_embedding.transpose(-1, -2).reshape(B, C, H, W))
        answer_bags = prompt[answer_mask].reshape(
            B, -1, len(self.nt_ans), C
        )  ## B, m, nt_ans, C

        prediction = {}

        for i, k in enumerate(self.nt_ans):
            q = self.heads[k](answer_bags[:, :, i, :])

            if k == "mask":
                mask = q @ vision.flatten(-2, -1)
                mask = mask.reshape(B, -1, H * 4, W * 4)
                prediction[k] = F.interpolate(
                    mask, size=ori_size, mode="bilinear", align_corners=True
                )  ## B, m, h, w
            elif k == "prob":
                prediction[k] = q.squeeze(-1)  ## B, m, 1
            elif k == "label":
                prediction[k] = q  ## B, m, n_cate

            assert len(q) == 1, "handle one by one"
            prediction[k] = prediction[k].squeeze(0)

        return prediction


class ProSRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config.transformer_width

        self.vision_tower = Tower(
            tower_name=config.vision_tower,
            tower_width=config.vision_tower_width,
            d_model=d_model,
            channel_side_attn=config.vision_tower_tokens,
        )

        self.trajectory_tower = Tower(
            tower_name=config.trajectory_tower,
            tower_width=config.trajectory_tower_width,
            d_model=d_model,
            channel_side_attn=config.trajectory_tower_tokens,
        )

        self.transformer = Transformer(
            width=d_model,
            layers=config.transformer_layers,
            heads=config.transformer_heads,
            add_pe=True,
        )

        self.mask_decoder = MaskDecoder(config=config)

        self.vocab = Vocab(config)

        self.register_buffer(
            "pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False
        )

        self.config = config

    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_input):
        """
        batched_input is a list of dict:
          'image': The image as a torch tensor in 3xHxW format,
            already transformed for input to the model.
          'original_size': (tuple(int, int)) The original size of
            the image before transformation, as (H, W).
        """
        prediction = []
        loss = {}
        batch_size = len(batched_input)

        for x in batched_input:
            prompt, answer_mask, vision_embedding = self.get_prompt(x)
            prompt = self.transformer(prompt)
            out = self.mask_decoder(
                prompt,
                answer_mask,
                vision_embedding,
                ori_size=(x["height"], x["width"]),
            )

            if True:
                mini_calc = self.calc(out, x)
                for k, v in mini_calc["loss"].items():
                    loss[k] = loss.get(k, 0.0) + v / batch_size * 1.0

            if True:
                out["mask"] = out["mask"].sigmoid()
                out["label"] = torch.argmax(out["label"], dim=-1)
                out["prob"] = torch.sigmoid(out["prob"])
                prediction.append(out)

        return {"loss": loss, "prediction": prediction}

    def matching(self, pred, gt):
        gt_masks = gt["gt_masks"].detach()
        pred_masks = torch.repeat_interleave(
            pred["mask"][:, None, :, :], len(gt_masks), dim=1
        ).detach()  ## m, m, H, W
        cost = []
        for i in range(len(gt_masks)):
            cost.append(
                batch_mask_loss(
                    preds=pred_masks[i],
                    targets=gt_masks,
                    ce_loss_weight=self.config.ce_loss_weight,
                    dice_loss_weight=self.config.dice_loss_weight,
                )
                .cpu()
                .numpy()
            )
        cost = np.stack(cost, axis=0)  ## m, m
        i_i, j_j = scipy.optimize.linear_sum_assignment(cost)

        assert np.array_equal(i_i, np.arange(len(i_i)))

        return i_i, j_j

    def calc(self, pred, gt):
        """
        pred: dict of m,(1/H,W/n_cate)
        """
        i_i, j_j = self.matching(pred, gt)

        gt_masks = gt["gt_masks"][j_j]
        gt_labels = gt["gt_labels"][j_j]
        gt_probs = gt["gt_probs"][j_j]

        ## calc_loss
        loss_mask = (
            batch_mask_loss_in_points(
                preds=pred["mask"],
                targets=gt_masks,
                ce_loss_weight=self.config.ce_loss_weight,
                dice_loss_weight=self.config.dice_loss_weight,
                K=self.config.K_points,
            ).mean()
            * self.config.mask_weight
        )

        loss_label = (
            F.cross_entropy(pred["label"], gt_labels) * self.config.label_weight
        )

        loss_prob = (
            F.l1_loss(torch.sigmoid(pred["prob"]), gt_probs) * self.config.prob_weight
        )
        
        ## metrics
        pass

        return {
            "loss": {
                "loss_mask": loss_mask,
                "loss_label": loss_label,
                "loss_prob": loss_prob,
            },
            "metrics": {
                
            }
        }

    def get_prompt(self, x: dict) -> torch.Tensor:
        input_image = self.preprocess(x["image"].unsqueeze(0))  # 1, 3, h, w
        vision_tokens, vision_embedding = self.vision_tower(input_image)  # 1, L, C

        condition_images = self.preprocess(x["condition_images"])  # n, 3, h, w

        tk_label = self.vocab("label", x["condition_labels"])  # 1, n, C

        if len(condition_images) > 0:
            tk_cond, _ = self.trajectory_tower(condition_images)  # n, L, C
        else:
            n_ttt = self.config.trajectory_tower_tokens
            tk_cond = torch.empty(
                (0, n_ttt, vision_tokens.shape[2]),
                device=self.device(),
                dtype=vision_tokens.dtype,
            )

        condition_tokens = torch.cat(
            [tk_label.transpose(0, 1), tk_cond], dim=1
        )  # n, 1+L, C
        condition_tokens = condition_tokens.flatten(0, 1).unsqueeze(0)  # 1, n(1+L), C

        answer_tokens = self.vocab(
            "answer", torch.arange(x["n_pred"]).long().to(self.device())
        )  # 1, a*4, C

        pt = [
            ("im_st", self.vocab("im_st")),
            ("image", vision_tokens),
            ("im_end", self.vocab("im_end")),
            ("cond_st", self.vocab("cond_st")),
            ("condition", condition_tokens),
            ("cond_end", self.vocab("cond_end")),
            ("answer_st", self.vocab("answer_st")),
            ("answer", answer_tokens),
            ("answer_end", self.vocab("answer_end")),
        ]

        prompt = torch.cat([v for k, v in pt], dim=1)

        answer_mask = [
            {True: torch.ones_like, False: torch.zeros_like}[k == "answer"](v[:, :, 0])
            for k, v in pt
        ]
        answer_mask = torch.cat(answer_mask, dim=1).gt(0.5)

        return prompt, answer_mask, vision_embedding
        # 1, full_length, C | 1, full_length | 1, 1024(Length), C

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """x: *, 3, h, w
        Normalize pixel values and resize to a square input.
        """
        x = (x - self.pixel_mean[None, :, :, :]) / self.pixel_std[None, :, :, :]
        size = (self.config.tower_image_size, self.config.tower_image_size)
        return F.interpolate(x, size=size, mode="bilinear")
