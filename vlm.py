import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel, AutoModelForCausalLM


class MinimoProjector(nn.Module):
    """
    Small bridge that maps image features into the text model hidden space.

    SigLIP emits vectors in its own embedding space, while Minimo expects token
    embeddings shaped for the language model hidden size. The projector learns a
    translation between those two spaces.
    """

    def __init__(self, in_features=768, out_features=896):
        super().__init__()

        # A two-layer MLP is simple, expressive, and much cheaper than trying to
        # train a full cross-modal transformer from scratch. The hidden width is
        # set to the output width so the projection stays lightweight.
        self.proj = nn.Sequential(
            nn.Linear(in_features, out_features, bias=False),
            nn.GELU(),
            nn.Linear(out_features, out_features, bias=False),
        )

    def forward(self, x):
        return self.proj(x)


class MinimoVLM(nn.Module):
    """
    Combine the local language model with a pretrained vision encoder.

    The design keeps the text model responsible for generation and uses SigLIP
    only as a feature extractor. That is a practical architecture for a local
    project because the expensive vision knowledge comes from a pretrained model
    instead of requiring multimodal pretraining from scratch.
    """

    def __init__(self, hf_model_path="checkpoints/hf_minimo_base", vision_model_name="google/siglip-base-patch16-224"):
        super().__init__()

        print(f"Loading Hugging Face Minimo from: {hf_model_path}")
        self.llm = AutoModelForCausalLM.from_pretrained(hf_model_path)

        print(f"Loading vision encoder: {vision_model_name}")
        self.vision_encoder = AutoModel.from_pretrained(vision_model_name).vision_model
        self.vision_processor = AutoImageProcessor.from_pretrained(vision_model_name)

        vision_hidden_size = self.vision_encoder.config.hidden_size
        self.projector = MinimoProjector(
            in_features=vision_hidden_size,
            out_features=self.llm.config.hidden_size,
        )

    def generate_with_image_and_rag(self, image, text_query, retrieved_rag_context, tokenizer):
        """
        Generate a response conditioned on image features and retrieved text.

        The image patches are embedded first, the projector maps those features
        into the language-model space, and then the projected image embeddings
        are concatenated with ordinary text embeddings before generation.
        """
        pixel_values = self.vision_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.llm.device)

        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_embeds = vision_outputs.last_hidden_state
        projected_image_embeds = self.projector(image_embeds)

        full_text = f"Context: {retrieved_rag_context}\nUser: {text_query}\nMinimo:"
        text_tokens = tokenizer.encode(full_text).ids
        text_tokens_tensor = torch.tensor([text_tokens], dtype=torch.long, device=self.llm.device)

        # Accessing the token embedding layer directly makes it possible to mix
        # image-derived vectors and text-derived vectors inside one shared input
        # sequence for generation.
        text_embeds = self.llm.model.embed_tokens(text_tokens_tensor)

        combined_embeds = torch.cat([projected_image_embeds, text_embeds], dim=1)

        print("Generating response...")
        output_ids = self.llm.generate(
            inputs_embeds=combined_embeds,
            # `150` is long enough for a useful answer while still keeping local
            # latency manageable on a modest machine.
            max_new_tokens=150,
            # `0.7` adds some variety without making the model drift too wildly.
            temperature=0.7,
            do_sample=True,
        )

        return tokenizer.decode(output_ids[0].tolist())
