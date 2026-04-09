import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

class MinimoProjector(nn.Module):
    """
    2-Layer MLP vision projector to map visual embeddings into 
    the language model’s dimension space.
    """
    def __init__(self, in_features, out_features=768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, out_features, bias=False),
            nn.GELU(),
            nn.Linear(out_features, out_features, bias=False)
        )

    def forward(self, x):
        return self.proj(x)

class MinimoVLM(nn.Module):
    """
    Vision-Language Model integrating SigLIP vision encoder 
    and the base Minimo Causal LM.
    """
    def __init__(self, llm_model, vision_model_name="google/siglip-base-patch16-224"):
        super().__init__()
        self.llm = llm_model
        
        print(f"Loading vision encoder: {vision_model_name}")
        # Note: In a real training scenario, you might want to freeze the vision model
        self.vision_encoder = AutoModel.from_pretrained(vision_model_name).vision_model
        self.vision_processor = AutoImageProcessor.from_pretrained(vision_model_name)
        
        # SigLIP base hidden size is 768
        vision_hidden_size = self.vision_encoder.config.hidden_size
        self.projector = MinimoProjector(in_features=vision_hidden_size, out_features=768)

    def forward(self, pixel_values, text_tokens=None):
        """
        Extract visual features and project them to LLM space.
        """
        # 1. Get vision embeddings
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        # We usually take the pooler output or the final hidden state sequence
        image_embeds = vision_outputs.last_hidden_state  # [B, NumPatches, VisionDim]
        
        # 2. Project to LLM space
        projected_embeds = self.projector(image_embeds)  # [B, NumPatches, 768]
        
        # In a full forward pass, you would prepend `projected_embeds` to the 
        # text token embeddings in `self.llm` and pass through the Transformer blocks.
        # This is a structural placeholder for integration.
        return projected_embeds
