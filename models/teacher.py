import torch
from transformers import AutoModelForCausalLM, Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from deepseek_vl2.models import DeepseekVLV2Processor
import copy
import gc

# ====================================== #
#          DEEPSEEK-VL2 MODELS           #
# ====================================== #

class LLMFeatureExtractor(torch.nn.Module):
    def __init__(self, conversation_template, model_name="deepseek-ai/deepseek-vl2-tiny", layer1_idx=0, layer2_idx=-1):
        super().__init__()

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = DeepseekVLV2Processor.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True
        ).to(torch.bfloat16).to(self.device).eval()

        num_layers = self.model.config.language_config.num_hidden_layers
        total_states = num_layers + 1
        assert -total_states <= layer1_idx < total_states
        assert -total_states <= layer2_idx < total_states
        self.layer1_idx = layer1_idx
        self.layer2_idx = layer2_idx
        self.embed_dim = self.model.config.language_config.hidden_size
        self.num_global_patches = 14 * 14

        self.conversation_template = conversation_template

    @torch.no_grad()
    def forward(self, pil_images):
        processed_outputs = []
        for image in pil_images:
            conversation = copy.deepcopy(self.conversation_template)
            processed_output = self.processor.process_one(
                conversations=conversation,
                images=[image]
            )
            processed_outputs.append(processed_output)

        prepare_inputs = self.processor.batchify(processed_outputs).to(self.device)

        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs['attention_mask'],
            output_hidden_states=True,
            use_cache=False
        )

        hidden_states = outputs.hidden_states
        features1 = hidden_states[self.layer1_idx]
        features2 = hidden_states[self.layer2_idx]

        image_seq_mask = prepare_inputs['images_seq_mask'][0]

        all_visual_embeds1 = features1[:, image_seq_mask, :]
        all_visual_embeds2 = features2[:, image_seq_mask, :]
        input_embeds_visual = hidden_states[0][:, image_seq_mask, :]

        newline_embedding = self.model.image_newline
        separator_embedding = self.model.view_seperator

        is_newline_token = torch.all(input_embeds_visual == newline_embedding, dim=2)
        is_separator_token = torch.all(input_embeds_visual == separator_embedding, dim=2)

        pure_patch_mask_1d = ~(is_newline_token[0] | is_separator_token[0])

        pure_embeddings1 = all_visual_embeds1[:, pure_patch_mask_1d, :]
        pure_embeddings2 = all_visual_embeds2[:, pure_patch_mask_1d, :]

        final_global_view1 = pure_embeddings1[:, :self.num_global_patches, :]
        final_global_view2 = pure_embeddings2[:, :self.num_global_patches, :]

        return final_global_view1, final_global_view2
    
class ViTFeatureExtractor(torch.nn.Module):
    def __init__(self, layers, model_name="deepseek-ai/deepseek-vl2-tiny"):
        super().__init__()
        
        self.layers = layers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        full_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        self.fe = full_model.vision
        self.fe.to(self.device).eval()

        self.patch_size = self.fe.patch_embed.patch_size[0]
        self.embed_dim = self.fe.embed_dim

        del full_model.language
        del full_model.projector
        del full_model
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(self, x):
        x = x.to(self.device, dtype=torch.bfloat16)
        return self.fe.get_intermediate_layers(x, n=self.layers)

# ====================================== #
#            QWEN2-VL MODELS             #
# ====================================== #
    
class QwenViTFeatureExtractor(torch.nn.Module):
    def __init__(self, model, processor, layers=[20, 24]):
        super().__init__()
        self.full_model = model
        self.processor = processor
        self.device = model.device

        self.layers = sorted(layers)
        self.visual = self.full_model.visual
        self.extracted_features = {}

        def get_activation(name):
            def hook(model, input, output):
                self.extracted_features[name] = output.detach()
            return hook
        
        for i in self.layers:
            self.visual.blocks[i].register_forward_hook(get_activation(f"layer_{i}"))

    def _unshuffle_patches(self, features, batch_size, grid_h, grid_w, merge_size):
        features = features.view(batch_size, -1, features.shape[-1])
        features = features.view(batch_size, grid_h, grid_w, merge_size, merge_size, -1)
        features = features.permute(0, 1, 3, 2, 4, 5).contiguous()
        h_patches = grid_h * merge_size
        w_patches = grid_w * merge_size
        features = features.view(batch_size, h_patches, w_patches, -1)
        return features

    @torch.no_grad()
    def forward(self, pil_images):
        batch_size = len(pil_images)

        inputs = self.processor.image_processor(
            images=pil_images,
            return_tensors="pt"
        )

        pixel_values = inputs.pixel_values.to(self.device, dtype=torch.bfloat16)
        image_grid_thw = inputs.image_grid_thw.to(self.device)
        _ = self.visual(hidden_states=pixel_values, grid_thw=image_grid_thw)
        
        feat_early_raw = self.extracted_features[f"layer_{self.layers[0]}"]
        feat_late_raw = self.extracted_features[f"layer_{self.layers[1]}"]

        _, h_patches, w_patches = image_grid_thw[0].tolist()
        merge_size = self.visual.spatial_merge_size
        grid_h = h_patches // merge_size
        grid_w = w_patches // merge_size
        
        batch_early = self._unshuffle_patches(feat_early_raw, batch_size, grid_h, grid_w, merge_size)
        batch_late = self._unshuffle_patches(feat_late_raw, batch_size, grid_h, grid_w, merge_size)

        return batch_early, batch_late

class QwenLLMFeatureExtractor(torch.nn.Module):
    def __init__(self, model, processor, layers=[21, 25]):
        super().__init__()
        self.model = model
        self.processor = processor
        self.device = model.device

        self.layers = sorted(layers)
        
        self.image_token_id = self.model.config.image_token_id
        self.merge_size = self.model.config.vision_config.spatial_merge_size

    @torch.no_grad()
    def forward(self, pil_images, conversation_template):
        if not isinstance(pil_images, list):
            pil_images = [pil_images]
        batch_size = len(pil_images)

        text_prompts = [
            self.processor.apply_chat_template(conversation_template, add_generation_prompt=False)
            for _ in range(batch_size)
        ]

        inputs = self.processor(
            text=text_prompts, images=pil_images, padding=False, return_tensors="pt"
        ).to(self.device)

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True
        )

        feat_early_raw = outputs.hidden_states[self.layers[0]]
        feat_late_raw = outputs.hidden_states[self.layers[1]]

        image_mask = inputs.input_ids == self.image_token_id
        
        _, h_vit, w_vit = inputs.image_grid_thw[0].tolist()
        
        h_llm = h_vit // self.merge_size
        w_llm = w_vit // self.merge_size

        earlier_feat = feat_early_raw[image_mask].view(batch_size, h_llm, w_llm, -1)
        later_feat = feat_late_raw[image_mask].view(batch_size, h_llm, w_llm, -1)

        return earlier_feat, later_feat