"""
ComfyUI Batch Prompt & LoRA Testing Extension
Allows testing multiple prompts with multiple LoRAs automatically
"""

class PromptLoraTestBench:
    """
    Node that generates all combinations of prompts and LoRAs
    """

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        import os

        # Get LoRA directories
        lora_paths = folder_paths.get_folder_paths("loras")

        # Collect all subdirectories
        subdirs = ["(use lora_names field)"]
        for base_path in lora_paths:
            if os.path.exists(base_path):
                # Add base directory
                subdirs.append(base_path)
                # Add all subdirectories
                for root, dirs, files in os.walk(base_path):
                    for d in dirs:
                        subdirs.append(os.path.join(root, d))

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "prompts": ("STRING", {
                    "multiline": True,
                    "default": "a cat\na dog\na bird",
                    "dynamicPrompts": False
                }),
                "negative_prompts": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": False
                }),
                "lora_names": ("STRING", {
                    "multiline": True,
                    "default": "lora1\nlora2",
                    "dynamicPrompts": False
                }),
                "lora_directory": (subdirs, {
                    "default": "(use lora_names field)"
                }),
                "lora_strength_model": ("FLOAT", {
                    "default": 1.0,
                    "min": -20.0,
                    "max": 20.0,
                    "step": 0.01
                }),
                "lora_strength_clip": ("FLOAT", {
                    "default": 1.0,
                    "min": -20.0,
                    "max": 20.0,
                    "step": 0.01
                }),
            },
            "optional": {
                "prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "generate_combinations"
    CATEGORY = "testing"

    def generate_combinations(self, model, clip, prompts, negative_prompts, lora_names,
                            lora_directory, lora_strength_model, lora_strength_clip,
                            prompt=None, negative_prompt=None):
        import os
        import folder_paths
        import re

        # Use override inputs if provided
        if prompt is not None:
            prompts = prompt
        if negative_prompt is not None:
            negative_prompts = negative_prompt

        # Parse inputs
        prompt_list = [p.strip() for p in prompts.strip().split('\n') if p.strip()]
        negative_list = [n.strip() for n in negative_prompts.strip().split('\n') if n.strip()] if negative_prompts else []

        # Check if we should use directory or lora_names
        if lora_directory and lora_directory != "(use lora_names field)":
            # Scan directory for LoRA files
            lora_list = []
            if os.path.exists(lora_directory):
                for file in os.listdir(lora_directory):
                    if file.endswith(('.safetensors', '.ckpt', '.pt', '.bin')):
                        lora_list.append(file)
                lora_list.sort()  # Sort alphabetically
        else:
            # Use lora_names field
            lora_list = [l.strip() for l in lora_names.strip().split('\n') if l.strip()]

        # If no negative prompts, use empty string for all
        if not negative_list:
            negative_list = [""] * len(prompt_list)
        # If fewer negative prompts than positive, repeat the last one
        elif len(negative_list) < len(prompt_list):
            negative_list.extend([negative_list[-1]] * (len(prompt_list) - len(negative_list)))

        # If no prompts provided, use empty string
        if not prompt_list:
            prompt_list = [""]
            negative_list = [""]

        models_out = []
        clips_out = []

        # Get LoRA paths
        lora_paths = folder_paths.get_folder_paths("loras")

        def find_lora_file(lora_name):
            """Search for LoRA file in main folder and subfolders"""
            # Add extensions if not present
            possible_names = [lora_name]
            if not lora_name.endswith(('.safetensors', '.ckpt', '.pt', '.bin')):
                possible_names.extend([
                    f"{lora_name}.safetensors",
                    f"{lora_name}.ckpt",
                    f"{lora_name}.pt",
                    f"{lora_name}.bin"
                ])

            for base_path in lora_paths:
                for name in possible_names:
                    # Try direct path first
                    direct_path = os.path.join(base_path, name)
                    if os.path.exists(direct_path):
                        return direct_path

                    # Search in subfolders
                    for root, dirs, files in os.walk(base_path):
                        if name in files:
                            return os.path.join(root, name)
            return None

        # Parse LoRA entries to extract name and strength
        def parse_lora_entry(lora_entry):
            """Parse LoRA entry in format <lora:name:strength> or just name"""
            pattern = r'<lora:([^:>]+):([0-9.]+)>'
            match = re.match(pattern, lora_entry)
            if match:
                return match.group(1), float(match.group(2))
            else:
                return lora_entry, None  # Use default strength

        # If no LoRAs, just return models and clips for each prompt
        if not lora_list:
            for i, pos_prompt in enumerate(prompt_list):
                models_out.append(model)
                clips_out.append(clip)
            return (models_out, clips_out)

        # Generate all combinations: prompts × LoRAs
        for lora_entry in lora_list:
            lora_name, custom_strength = parse_lora_entry(lora_entry)

            # If using directory, construct full path directly
            if lora_directory and lora_directory != "(use lora_names field)":
                lora_file_path = os.path.join(lora_directory, lora_name)
                if not os.path.exists(lora_file_path):
                    lora_file_path = None
            else:
                # Find LoRA file in folder or subfolders
                lora_file_path = find_lora_file(lora_name)

            if lora_file_path is None:
                print(f"Warning: LoRA file not found: {lora_name}")
                continue

            # Load the LoRA
            import comfy.utils
            import comfy.sd

            lora = comfy.utils.load_torch_file(lora_file_path, safe_load=True)

            # Use custom strength if provided, otherwise use default
            strength_model = custom_strength if custom_strength is not None else lora_strength_model
            strength_clip = custom_strength if custom_strength is not None else lora_strength_clip

            model_lora, clip_lora = comfy.sd.load_lora_for_models(
                model, clip,
                lora,
                strength_model,
                strength_clip
            )

            # For each prompt with this LoRA
            for i, pos_prompt in enumerate(prompt_list):
                neg_prompt = negative_list[i] if i < len(negative_list) else ""

                # Encode prompts with the LoRA-modified CLIP
                tokens = clip_lora.tokenize(pos_prompt)
                cond, pooled = clip_lora.encode_from_tokens(tokens, return_pooled=True)

                neg_tokens = clip_lora.tokenize(neg_prompt)
                neg_cond, neg_pooled = clip_lora.encode_from_tokens(neg_tokens, return_pooled=True)

                models_out.append(model_lora)
                clips_out.append(clip_lora)

        return (models_out, clips_out)


class TextCombiner:
    """
    Utility node to combine prompt info with LoRA name for filename generation
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
                "lora_name": ("STRING", {"forceInput": True}),
                "separator": ("STRING", {"default": "_"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "combine"
    CATEGORY = "testing"

    def combine(self, prompt, lora_name, separator="_"):
        # Clean strings for filename
        import re
        clean_prompt = re.sub(r'[^\w\s-]', '', prompt)[:50]
        clean_prompt = re.sub(r'[\s]+', '_', clean_prompt)
        clean_lora = lora_name.replace('.safetensors', '').replace('.ckpt', '')

        return (f"{clean_lora}{separator}{clean_prompt}",)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "PromptLoraTestBench": PromptLoraTestBench,
    "TextCombiner": TextCombiner,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptLoraTestBench": "Prompt & LoRA Test Bench",
    "TextCombiner": "Text Combiner",
}
