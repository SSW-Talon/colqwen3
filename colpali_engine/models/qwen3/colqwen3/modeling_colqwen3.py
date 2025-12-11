from typing import ClassVar
import torch
import torch.nn as nn
from transformers.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLConfig
)

class ColQwen3(Qwen3VLForConditionalGeneration):
    """
    ColQwen3 model implementation for ColPali-style late interaction.
    Based on Qwen3-VL backbone.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"

    _checkpoint_conversion_mapping = {
        "^model.language_model.model": "model.model.language_model",
        "^model.vision_tower": "model.model.vision_tower",
        "^model.multi_modal_projector": "model.model.multi_modal_projector",
        "^model.language_model.lm_head": "model.lm_head",
    }

    def __init__(self, config: Qwen3VLConfig, mask_non_image_embeddings: bool = False):
        super().__init__(config=config)

        # --------------------------
        # ❗ 修复 hidden_size 读取方式
        # --------------------------
        self.hidden_size = self.config.text_config.hidden_size  # Qwen3-VL true hidden size

        # Remove LM head (ColPali does not need generation)
        self.model.lm_head = torch.nn.Identity()

        # ColPali embedding dim
        self.dim = 128

        # ColBERT-like projection layer
        self.custom_text_proj = nn.Linear(self.hidden_size, self.dim)

        self.padding_side = "left"
        self.mask_non_image_embeddings = mask_non_image_embeddings

        self.post_init()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ):
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )

    def forward(self, *args, **kwargs):
        # Handle pixel_values unpadding from ColQwen3Processor
        if "pixel_values" in kwargs:
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]
            kwargs["pixel_values"] = torch.cat(
                [seq[:offset] for seq, offset in zip(kwargs["pixel_values"], offsets)],
                dim=0,
            )

        kwargs.pop("return_dict", True)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)

        outputs = self.model(*args, output_hidden_states=True, return_dict=True, **kwargs)
        last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)

        # ---- Projection to 128-D ----
        proj = self.custom_text_proj(last_hidden)
        proj = proj / proj.norm(dim=-1, keepdim=True)

        # Apply attention mask
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)

        # Optional masking: only keep image embeddings
        if "pixel_values" in kwargs and self.mask_non_image_embeddings:
            image_mask = (kwargs["input_ids"] == self.config.image_token_id).unsqueeze(-1)
            proj = proj * image_mask

        return proj

    @property
    def patch_size(self):
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self):
        return self.visual.config.spatial_merge_size
