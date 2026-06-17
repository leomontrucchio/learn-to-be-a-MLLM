# ====================================== #
#            FE FOR QWEN2-VL             #
# ====================================== #
    
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from PIL import Image

class QwenViTFeatureExtractor(torch.nn.Module):
    def __init__(self, model, processor, layers=[20, 24]):
        super().__init__()
        self.full_model = model
        self.processor = processor
        self.device = model.device

        self.layers = sorted(layers)
        self.visual = self.full_model.visual
        self.extracted_features = {}

        # Hook to register internal features
        def get_activation(name):
            def hook(model, input, output):
                self.extracted_features[name] = output.detach()
            return hook
        
        for i in self.layers:
            self.visual.blocks[i].register_forward_hook(get_activation(f"layer_{i}"))

    def _unshuffle_patches(self, features, batch_size, grid_h, grid_w, merge_size):
        """
        Method to convert patches order from the one for pixel shuffle to the one of the
        original image (for anomaly maps).
        Convert from [Batch*Seq, Dim] to [Batch, H, W, Dim]
        """
        # View in batches
        features = features.view(batch_size, -1, features.shape[-1])
        
        # 2x2 blocks reconstruction
        features = features.view(batch_size, grid_h, grid_w, merge_size, merge_size, -1)
        
        # Permute for Raster Scan order
        features = features.permute(0, 1, 3, 2, 4, 5).contiguous()
        
        # Final flattening to 2D spatial grid
        h_patches = grid_h * merge_size
        w_patches = grid_w * merge_size
        features = features.view(batch_size, h_patches, w_patches, -1)
        
        return features

    @torch.no_grad()
    def forward(self, pil_images):
        """
        Args:
            pil_images: List of PIL Images.
        """
        batch_size = len(pil_images)

        # 1. Images Processing
        inputs = self.processor.image_processor(
            images=pil_images,
            return_tensors="pt"
        )

        # 2. Forward with hooks
        pixel_values = inputs.pixel_values.to(self.device, dtype=torch.bfloat16)
        image_grid_thw = inputs.image_grid_thw.to(self.device)
        _ = self.visual(hidden_states=pixel_values, grid_thw=image_grid_thw)
        
        feat_early_raw = self.extracted_features[f"layer_{self.layers[0]}"]
        feat_late_raw = self.extracted_features[f"layer_{self.layers[1]}"]

        # 3. Compute grid parameters
        _, h_patches, w_patches = image_grid_thw[0].tolist()
        merge_size = self.visual.spatial_merge_size
        grid_h = h_patches // merge_size
        grid_w = w_patches // merge_size
        
        # 4. Reorder patches in Raster Scan order
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
        """
        Args:
            pil_images: List of PIL images.
        Returns:
            earlier_feat: [Batch, H_llm, W_llm, Dim]
            later_feat:   [Batch, H_llm, W_llm, Dim]
        """
        if not isinstance(pil_images, list):
            pil_images = [pil_images]
        batch_size = len(pil_images)

        # 1. Prompts creation
        text_prompts = [
            self.processor.apply_chat_template(conversation_template, add_generation_prompt=False)
            for _ in range(batch_size)
        ]

        # 2. Input preprocessing
        inputs = self.processor(
            text=text_prompts, images=pil_images, padding=False, return_tensors="pt"
        ).to(self.device)

        # 3. Forward pass with hidden states extraction
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True
        )

        # 4. Specified hidden layers extraction
        feat_early_raw = outputs.hidden_states[self.layers[0]]
        feat_late_raw = outputs.hidden_states[self.layers[1]]

        # 5. Visual token identification and Reshape
        image_mask = inputs.input_ids == self.image_token_id
        
        # Recall grid geometry
        _, h_vit, w_vit = inputs.image_grid_thw[0].tolist()
        
        h_llm = h_vit // self.merge_size
        w_llm = w_vit // self.merge_size

        # Extraction and Final reshape
        earlier_feat = feat_early_raw[image_mask].view(batch_size, h_llm, w_llm, -1)
        later_feat = feat_late_raw[image_mask].view(batch_size, h_llm, w_llm, -1)

        return earlier_feat, later_feat
    
    

# ====================================== #
#             PIPELINE DI TEST           #
# ====================================== #

def predict_qwen_shape(orig_h, orig_w):
    FACTOR = 28 # 14 * 2
    new_h, new_w = round(orig_h / FACTOR) * FACTOR, round(orig_w / FACTOR) * FACTOR
    return new_h, new_w, new_h // 14, new_w // 14


if __name__ == "__main__":
    MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
    
    # 0. Inizializzazione Modello Unico (Shared)
    print(f"--- Caricamento modello condiviso: {MODEL_NAME} ---")
    shared_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
    ).eval()
    shared_processor = Qwen2VLProcessor.from_pretrained(MODEL_NAME)

    # 1. Test Unshuffle (Logica)
    print("\n=== TEST 1: Verifica Logica Matematica (Unshuffle) ===")
    qwen_shuffled_indices = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
    fake_features = torch.tensor(qwen_shuffled_indices).float().view(1, 16, 1)
    extractor_vit = QwenViTFeatureExtractor(shared_model, shared_processor)
    unshuffled = extractor_vit._unshuffle_patches(fake_features, 1, 2, 2, 2)
    if unshuffled.flatten().tolist() == list(range(16)):
        print("SUCCESS: Logica unshuffle corretta.")
    else: print("FAILURE: Ordine errato.")

    # 2. Test ViT con Immagine Reale
    print("\n=== TEST 2: Verifica ViT (Dimensioni) ===")
    W_ORIG, H_ORIG = 1600, 1280
    dummy_image = Image.new('RGB', (W_ORIG, H_ORIG), color='blue')
    exp_h_px, exp_w_px, exp_h_feat, exp_w_feat = predict_qwen_shape(H_ORIG, W_ORIG)
    feat_early_vit, _ = extractor_vit([dummy_image])
    if feat_early_vit.shape[1:3] == (exp_h_feat, exp_w_feat):
        print(f"SUCCESS: ViT ha generato {feat_early_vit.shape[2]}x{feat_early_vit.shape[1]} patches.")
    else: print("FAILURE: Mismatch dimensioni ViT.")

    # 3. Test LLM (Allineamento)
    print("\n=== TEST 3: Verifica LLM (Allineamento) ===")
    template_standard = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Is this OK?"}]}]
    extractor_llm = QwenLLMFeatureExtractor(shared_model, shared_processor)
    llm_early, _ = extractor_llm([dummy_image], template_standard)
    if llm_early.shape[1:3] == (exp_h_feat // 2, exp_w_feat // 2):
        print("SUCCESS: LLM allineato correttamente (pooling 2x2).")
    else: print("FAILURE: Mismatch allineamento LLM.")

    # 4. TEST DI CAUSALITÀ (Obiettivo: Dimostrare che il futuro non influenza il passato)
    print("\n=== TEST 4: Verifica Attention Causalità ===")
    # Prompt A: Immagine seguita da poco testo
    conv_a = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "The industrial product shown in this image is defect-free. All components are present in the correct quantity and arrangement, satisfying all structural and logical quality constraints."}]}]
    # Prompt B: Immagine seguita da testo lungo
    conv_b = [{"role": "user", "content": [
        {"type": "image"}, 
        {"type": "text", "text": (
                "You are a senior Quality Assurance Engineer. Provide your official assessment of this breakfast box.\n\n"
                "Assistant: Technical Report: Golden Sample Breakfast Box\n\n"
                "Left Compartment:\n"
                "The left compartment contains exactly two tangerines and one nectarine.\n"
                "The tangerines have a vibrant orange color with a slightly glossy surface, indicating ripeness. They exhibit a smooth texture with minor natural imperfections typical of fresh citrus fruits.\n"
                "The nectarine has a gradient of colors from yellow to red, suggesting it is ripe and ready to eat. Its skin appears smooth but with slight dimples, which is characteristic of nectarines.\n\n"
                "Right Compartment:\n"
                "The right compartment is filled with a mix of cereals, banana chips, and almonds.\n"
                "The cereal mix consists of various grains, including oats and possibly other types of nuts or seeds. The colors range from light brown to golden, indicating a toasted or roasted preparation. The distribution is even, ensuring each bite will contain a balanced mix of ingredients.\n"
                "The banana chips are sliced into uniform pieces, showcasing a pale yellow color with some darker spots where they were exposed to air during drying. Their edges appear slightly curled, which is typical for dried bananas.\n"
                "Almonds are scattered throughout the mixture, adding a contrasting dark brown color. They are whole and unprocessed, providing a crunchy texture to complement the softer elements.\n\n"
                "Container Material:\n"
                "The container is made of white plastic, likely polystyrene, designed for single-use packaging. It has a rectangular shape with rounded corners and a clear partition separating the two compartments.\n"
                "The material appears clean and free of any defects, meeting hygiene standards required for food packaging.\n\n"
                "Lighting and Background:\n"
                "The image is taken under bright, even lighting, which enhances the visibility of the colors and textures of the food items. There are no harsh shadows, ensuring all details are clearly visible.\n"
                "The background is a solid black color, which contrasts sharply with the white container and colorful food items, making them stand out prominently.\n\n"
                "Visual Standard:\n"
                "The overall presentation adheres to a high standard of cleanliness and organization. Each item is neatly placed within its designated section, demonstrating attention to detail and adherence to quality control measures.\n"
                "The combination of fresh fruits and nutritious cereals reflects a balanced and appealing meal option, suitable for consumers seeking a healthy breakfast choice.\n\n"
                "This detailed analysis confirms that the golden sample breakfast box meets all specified manufacturing constraints and visual standards, serving as an exemplary model for future production."
            )}
    ]}]

    print("Esecuzione Forward con Prompt A...")
    _, feat_a = extractor_llm([dummy_image], conv_a)
    print("Esecuzione Forward con Prompt B...")
    _, feat_b = extractor_llm([dummy_image], conv_b)

    # Confronto matematico: se l'attenzione è causale, la differenza deve essere zero (o vicina alla precisione di bfloat16)
    max_diff = torch.abs(feat_a - feat_b).max().item()
    print(f"Differenza massima tra le feature dell'immagine: {max_diff}")

    if max_diff < 1e-5:
        print("SUCCESS: Le feature sono identiche! L'attenzione causale funziona.")
        print("Il testo aggiunto DOPO l'immagine non ha minimamente influenzato i token visivi.")
    else:
        print("FAILURE: Le feature sono diverse. Attenzione: il contesto futuro sta influenzando il passato.")

    print("\n=== TUTTI I TEST COMPLETATI ===")