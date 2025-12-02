from typing import ClassVar

import torch
import torch.nn as nn
from transformers.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    # Qwen3VLPreTrainedModel,
    Qwen3VLConfig
)

class ColQwen3(Qwen3VLForConditionalGeneration):
    """
    ColQwen3 model implementation, following the achitecture from the article "ColPali: Efficient Document Retrieval
    with Vision Language Models" paper. Based on the Qwen3-VL backbone.

    Args:
        config (Qwen3VLConfig): The model configuration.
        mask_non_image_embeddings (Optional[bool]): Whether to ignore all tokens embeddings
            except those of the image at inference.
            Defaults to False --> Do not mask any embeddings during forward pass.
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
        
        
        self.model.lm_head = torch.nn.Identity()
        
        self.dim = 128
        # qwen3-VL self.hidden_size = 2048
        
        self.custom_text_proj = nn.Linear(self.config.hidden_size,self.dim)
        # self.cust_custom_text_proj = nn.Linear(self.hidden_size,self.dim)
        
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
    
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Handle the custom "pixel_values" input obtained with `ColQwen3Processor`
        # Qwen3-VL 依然使用 flatten 的 visual tokens，需要根据 grid_thw 进行 unpadding
        if "pixel_values" in kwargs:
            # grid_thw shape: (batch, 3) -> (Time, Height, Width)
            # 对于文档图像，Time 通常为 1
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]  # (batch_size,)
            
            # 确保 pixel_values 在 device 上并且类型正确
            kwargs["pixel_values"] = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(kwargs["pixel_values"], offsets)],
                dim=0,
            )
        kwargs.pop("return_dict", True)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)
        
        # 获取基础模型的输出
        outputs = self.model(*args, output_hidden_states=True, return_dict=True, **kwargs)
        last_hidden_states = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)
        
        # 投影到低维空间 (128 dim for ColBERT/ColPali style)
        # L2 normalization
        proj = self.custom_text_proj(last_hidden_states)
        proj = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, dim)
        
        if "pixel_values" in kwargs and self.mask_non_image_embeddings:
            # Pools only the image embeddings
            image_mask = (kwargs["input_ids"] == self.config.image_token_id).unsqueeze(-1)
            proj = proj * image_mask
        return proj
    
    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size