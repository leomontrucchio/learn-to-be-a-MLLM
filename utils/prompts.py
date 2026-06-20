from utils.prompts_loco import PROMPTS_CONFIG as PROMPTS_LOCO
from utils.prompts_loco import PROMPTS_QWEN as PROMPTS_QWEN_LOCO
from utils.prompts_visa import PROMPTS_CONFIG as PROMPTS_VISA

def get_prompt(dataset, model_type, prompt_type, class_name=None):
    """
    Returns the appropriate prompt based on dataset, model_type, prompt_type and class_name.
    """
    if dataset == 'loco':
        if model_type == 'qwen':
            prompts = PROMPTS_QWEN_LOCO
        else:
            prompts = PROMPTS_LOCO
    elif dataset == 'visa':
        prompts = PROMPTS_VISA
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    # For generic prompts
    if prompt_type in ['just_image', 'generic', 'grounding', 'assistant_generic']:
        return prompts.get(prompt_type)
    
    # For class-specific prompts (assistant_u, assistant_s, grounding_s, distilled)
    specific_prompt_key = f"{prompt_type}_{class_name}"
    
    if specific_prompt_key in prompts:
        return prompts[specific_prompt_key]
    
    # Fallback if specific prompt is missing but generic exists
    print(f"Warning: Specific prompt '{specific_prompt_key}' not found. Falling back to generic if available.")
    return prompts.get(prompt_type)
