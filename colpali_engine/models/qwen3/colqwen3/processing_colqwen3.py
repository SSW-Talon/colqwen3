from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature
from transformers.models.qwen2_vl import Qwen2VLProcessor

from transformers.models.qwen3_vl import Qwen3VLProcessor


from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

class ColQwen3Processor(BaseVisualRetrieverProcessor, Qwen3VLProcessor):  
    """
    Processor for ColQwen3.

    Args:
        *args: Variable length argument list to be passed to the parent `Qwen3VLProcessor`.
        max_num_visual_tokens: The maximum number of visual tokens.
        **kwargs: Arbitrary keyword arguments.
    """
    
    visual_prompt_prefix: ClassVar[str] = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>"
    )
    query_augmentation_token: ClassVar[str] = "<|endoftext|>"
    image_token: ClassVar[str] = "<|image_pad|>"
    
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer.padding_side = "left"

    @classmethod
    def from_pretrained(
        cls,
        *args,
        device_map: Optional[str] = None,
        **kwargs,
    ):
        instance = super().from_pretrained(
            *args,
            device_map=device_map,
            **kwargs,
        )

        
        if "max_num_visual_tokens" in kwargs:
            instance.image_processor.max_pixels = kwargs["max_num_visual_tokens"] * 28 * 28
            instance.image_processor.size["longest_edge"] = instance.image_processor.max_pixels

        return instance
    
    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process images for ColQwen3.
        """

        images = [image.convert("RGB") for image in images]

        
        batch_doc = self(
            text=[self.visual_prompt_prefix] * len(images),
            images=images,
            padding="longest",
            return_tensors="pt",
        )

        
        offsets = batch_doc["image_grid_thw"][:, 1] * batch_doc["image_grid_thw"][:, 2]  # (batch_size,)

        # 将堆叠的 pixel_values 切分为列表
        pixel_values = list(
            torch.split(batch_doc["pixel_values"], offsets.tolist())
        )

        # Pad 到最大长度以便 batch 处理
        batch_doc["pixel_values"] = torch.nn.utils.rnn.pad_sequence(
            pixel_values, batch_first=True
        )  # (batch_size, max_num_patches, hidden/channels)

        return batch_doc
    
    def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
        """
        Process texts for ColQwen3.
        """
        return self(
            text=texts,
            return_tensors="pt",
            padding="longest",
        )

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like).
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        spatial_merge_size: int,
    ) -> Tuple[int, int]:
        """
        Calculate the grid dimensions (H, W) for Qwen3-VL.
        Qwen3-VL logic for resizing and merging is generally consistent with v2.5.
        """
        patch_size = self.image_processor.patch_size

        height_new, width_new = smart_resize(
            width=image_size[0],
            height=image_size[1],
            factor=patch_size * self.image_processor.merge_size,
            min_pixels=self.image_processor.size["shortest_edge"],
            max_pixels=self.image_processor.size["longest_edge"],
        )

        n_patches_x = width_new // patch_size // spatial_merge_size
        n_patches_y = height_new // patch_size // spatial_merge_size

        return n_patches_x, n_patches_y

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        return batch_images.input_ids == self.image_token_id