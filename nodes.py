"""ComfyUI Batch Prompt & LoRA Testing Extension"""

class PromptLoraTestBench:
    """Generates all combinations of prompts and LoRAs"""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        import os

        lora_paths = folder_paths.get_folder_paths("loras")
        subdirs = ["(use lora_names field)"]

        for base_path in lora_paths:
            if os.path.exists(base_path):
                subdirs.append(base_path)
                for root, dirs, _ in os.walk(base_path):
                    subdirs.extend(os.path.join(root, d) for d in dirs)

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
                "lora_range_start": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "tooltip": "Filter by number in filename (0 = disabled)"
                }),
                "lora_range_end": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "tooltip": "Filter by number in filename (0 = disabled)"
                }),
            },
            "optional": {
                "prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "CONDITIONING", "CONDITIONING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "positive", "negative", "lora_names", "prompts", "negative_prompts")
    OUTPUT_IS_LIST = (True, True, True, True, True, True, True)
    FUNCTION = "generate_combinations"
    CATEGORY = "testing"

    def generate_combinations(self, model, clip, prompts, negative_prompts, lora_names,
                            lora_directory, lora_strength_model, lora_strength_clip,
                            lora_range_start, lora_range_end,
                            prompt=None, negative_prompt=None):
        import os
        import folder_paths
        import re

        prompts = prompt if prompt is not None else prompts
        negative_prompts = negative_prompt if negative_prompt is not None else negative_prompts

        prompt_list = [p.strip() for p in prompts.strip().split('\n') if p.strip()]
        negative_list = [n.strip() for n in negative_prompts.strip().split('\n') if n.strip()] if negative_prompts else []

        lora_list = self._get_lora_list(lora_directory, lora_names)
        lora_list = self._apply_range_filter(lora_list, lora_range_start, lora_range_end)
        negative_list = self._normalize_negative_list(negative_list, len(prompt_list))

        if not prompt_list:
            prompt_list, negative_list = [""], [""]

        lora_paths = folder_paths.get_folder_paths("loras")
        outputs = {"models": [], "clips": [], "positive": [], "negative": [], "lora_names": [], "prompts": [], "negative_prompts": []}

        if not lora_list:
            return self._process_without_loras(model, clip, prompt_list, negative_list)

        for lora_entry in lora_list:
            lora_name, custom_strength = self._parse_lora_entry(lora_entry, re)
            lora_file_path = self._find_lora_path(lora_name, lora_directory, lora_paths, os)

            if not lora_file_path:
                print(f"Warning: LoRA file not found: {lora_name}")
                continue

            model_lora, clip_lora = self._load_lora(lora_file_path, model, clip,
                                                     custom_strength or lora_strength_model,
                                                     custom_strength or lora_strength_clip)

            for i, pos_prompt in enumerate(prompt_list):
                neg_prompt = negative_list[i] if i < len(negative_list) else ""
                self._add_outputs(outputs, model_lora, clip_lora, pos_prompt, neg_prompt, lora_name)

        return tuple(outputs.values())

    @staticmethod
    def _get_lora_list(lora_directory, lora_names):
        import os
        if lora_directory and lora_directory != "(use lora_names field)":
            if os.path.exists(lora_directory):
                lora_list = [f for f in os.listdir(lora_directory)
                           if f.endswith(('.safetensors', '.ckpt', '.pt', '.bin'))]
                return sorted(lora_list)
            return []
        return [l.strip() for l in lora_names.strip().split('\n') if l.strip()]

    @staticmethod
    def _apply_range_filter(lora_list, start, end):
        """Filter LoRA list by numbers found in filenames."""
        import re

        if start <= 0 and end <= 0:
            return lora_list

        filtered = []
        for lora_name in lora_list:
            # Extract all numbers from filename
            numbers = re.findall(r'\d+', lora_name)
            if numbers:
                # Use the largest number found (typically the step/epoch number)
                max_num = max(int(n) for n in numbers)

                # Check if within range
                start_ok = (start <= 0) or (max_num >= start)
                end_ok = (end <= 0) or (max_num <= end)

                if start_ok and end_ok:
                    filtered.append(lora_name)

        print(f"[LoRA Range Filter] Applied range {start}-{end}: {len(filtered)} of {len(lora_list)} LoRAs selected")
        return filtered

    @staticmethod
    def _normalize_negative_list(negative_list, prompt_count):
        if not negative_list:
            return [""] * prompt_count
        if len(negative_list) < prompt_count:
            negative_list.extend([negative_list[-1]] * (prompt_count - len(negative_list)))
        return negative_list

    @staticmethod
    def _parse_lora_entry(lora_entry, re):
        match = re.match(r'<lora:([^:>]+):([0-9.]+)>', lora_entry)
        return (match.group(1), float(match.group(2))) if match else (lora_entry, None)

    @staticmethod
    def _find_lora_path(lora_name, lora_directory, lora_paths, os):
        if lora_directory and lora_directory != "(use lora_names field)":
            path = os.path.join(lora_directory, lora_name)
            return path if os.path.exists(path) else None

        extensions = ('.safetensors', '.ckpt', '.pt', '.bin')
        possible_names = [lora_name]
        if not lora_name.endswith(extensions):
            possible_names.extend([f"{lora_name}{ext}" for ext in extensions])

        for base_path in lora_paths:
            for name in possible_names:
                direct_path = os.path.join(base_path, name)
                if os.path.exists(direct_path):
                    return direct_path

                for root, _, files in os.walk(base_path):
                    if name in files:
                        return os.path.join(root, name)
        return None

    @staticmethod
    def _load_lora(lora_file_path, model, clip, strength_model, strength_clip):
        import comfy.utils
        import comfy.sd
        lora = comfy.utils.load_torch_file(lora_file_path, safe_load=True)
        return comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)

    def _encode_conditioning(self, clip, positive, negative):
        pos_tokens = clip.tokenize(positive)
        pos_cond, pos_pooled = clip.encode_from_tokens(pos_tokens, return_pooled=True)

        neg_tokens = clip.tokenize(negative)
        neg_cond, neg_pooled = clip.encode_from_tokens(neg_tokens, return_pooled=True)

        return [[pos_cond, {"pooled_output": pos_pooled}]], [[neg_cond, {"pooled_output": neg_pooled}]]

    def _add_outputs(self, outputs, model, clip, positive, negative, lora_name):
        pos_cond, neg_cond = self._encode_conditioning(clip, positive, negative)
        outputs["models"].append(model)
        outputs["clips"].append(clip)
        outputs["positive"].append(pos_cond)
        outputs["negative"].append(neg_cond)
        outputs["lora_names"].append(lora_name)
        outputs["prompts"].append(positive)
        outputs["negative_prompts"].append(negative)

    def _process_without_loras(self, model, clip, prompt_list, negative_list):
        outputs = {"models": [], "clips": [], "positive": [], "negative": [], "lora_names": [], "prompts": [], "negative_prompts": []}
        for i, pos_prompt in enumerate(prompt_list):
            neg_prompt = negative_list[i] if i < len(negative_list) else ""
            self._add_outputs(outputs, model, clip, pos_prompt, neg_prompt, "no_lora")
        return tuple(outputs.values())


class ImageAnnotator:
    """Adds text annotations to images with LoRA names"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "lora_names": ("STRING", {"forceInput": True}),
                "enable_annotations": ("BOOLEAN", {"default": True}),
                "preserve_original": ("BOOLEAN", {"default": False}),
                "font_size": ("INT", {
                    "default": 18,
                    "min": 8,
                    "max": 128,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    INPUT_IS_LIST = (True, True, False, False, False)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "annotate_images"
    CATEGORY = "testing"

    def annotate_images(self, images, lora_names, enable_annotations, preserve_original, font_size):
        import torch
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        enable_annotations = enable_annotations[0] if isinstance(enable_annotations, list) else enable_annotations
        preserve_original = preserve_original[0] if isinstance(preserve_original, list) else preserve_original
        font_size = font_size[0] if isinstance(font_size, list) else font_size

        print(f"[ImageAnnotator] enable_annotations={enable_annotations}, preserve_original={preserve_original}")
        print(f"[ImageAnnotator] Processing {len(images)} images")

        if not enable_annotations:
            return (images,)

        font = self._load_font(ImageFont, font_size)
        output_images = []

        for img_tensor, lora_name in zip(images, lora_names):
            img_tensor_squeezed = img_tensor.squeeze(0) if img_tensor.dim() == 4 and img_tensor.shape[0] == 1 else img_tensor
            pil_img = self._tensor_to_pil(img_tensor_squeezed, np, Image)
            annotated_tensor = self._create_annotated_image(pil_img, lora_name, font, ImageDraw, np, torch)

            if preserve_original:
                output_images.extend([img_tensor, annotated_tensor])
            else:
                output_images.append(annotated_tensor)

        print(f"[ImageAnnotator] Returning {len(output_images)} images")
        return (output_images,)

    @staticmethod
    def _load_font(ImageFont, font_size):
        for font_name in ["arial.ttf", "DejaVuSans.ttf"]:
            try:
                return ImageFont.truetype(font_name, font_size)
            except:
                continue
        return ImageFont.load_default()

    @staticmethod
    def _tensor_to_pil(img_tensor, np, Image):
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np, mode='RGB')

    @staticmethod
    def _create_annotated_image(pil_img, text, font, ImageDraw, np, torch):
        annotated_img = pil_img.copy()
        draw = ImageDraw.Draw(annotated_img)

        bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

        padding = 5
        x, y = padding, pil_img.height - text_height - padding * 2

        draw.rectangle(
            [(x - padding, y - padding), (x + text_width + padding, y + text_height + padding)],
            fill="black"
        )
        draw.text((x, y), text, fill="white", font=font)

        annotated_np = np.array(annotated_img).astype(np.float32) / 255.0
        return torch.from_numpy(annotated_np).unsqueeze(0)


class ImageSaverWithMetadata:
    """Saves images with prompt and LoRA metadata embedded in PNG"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "optional": {
                "prompts": ("STRING", {"forceInput": True}),
                "negative_prompts": ("STRING", {"forceInput": True}),
                "lora_names": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    INPUT_IS_LIST = (True, False, True, True, True)
    CATEGORY = "testing"

    def save_images(self, images, filename_prefix, prompts=None, negative_prompts=None, lora_names=None):
        import torch
        import numpy as np
        from PIL import Image, PngImagePlugin
        import folder_paths
        import os
        import json

        filename_prefix = filename_prefix[0] if isinstance(filename_prefix, list) else filename_prefix
        output_dir = folder_paths.get_output_directory()

        # Create subfolder if prefix contains path separators
        if os.sep in filename_prefix or '/' in filename_prefix:
            filename_prefix = filename_prefix.replace('/', os.sep)
            subfolder = os.path.dirname(filename_prefix)
            filename_prefix = os.path.basename(filename_prefix)
            output_dir = os.path.join(output_dir, subfolder)
            os.makedirs(output_dir, exist_ok=True)

        results = []
        counter = self._get_next_counter(output_dir, filename_prefix)

        for idx, img_tensor in enumerate(images):
            # Convert tensor to PIL Image
            img_tensor_squeezed = img_tensor.squeeze(0) if img_tensor.dim() == 4 and img_tensor.shape[0] == 1 else img_tensor
            img_np = (img_tensor_squeezed.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np, mode='RGB')

            # Prepare metadata
            metadata = PngImagePlugin.PngInfo()

            if prompts and idx < len(prompts):
                metadata.add_text("prompt", prompts[idx])

            if negative_prompts and idx < len(negative_prompts):
                metadata.add_text("negative_prompt", negative_prompts[idx])

            if lora_names and idx < len(lora_names):
                metadata.add_text("lora", lora_names[idx])

            # Add combined metadata as JSON for easier parsing
            combined_metadata = {}
            if prompts and idx < len(prompts):
                combined_metadata["prompt"] = prompts[idx]
            if negative_prompts and idx < len(negative_prompts):
                combined_metadata["negative_prompt"] = negative_prompts[idx]
            if lora_names and idx < len(lora_names):
                combined_metadata["lora"] = lora_names[idx]

            if combined_metadata:
                metadata.add_text("parameters", json.dumps(combined_metadata, indent=2))

            # Save image
            filename = f"{filename_prefix}_{counter:05d}.png"
            filepath = os.path.join(output_dir, filename)
            pil_img.save(filepath, pnginfo=metadata, compress_level=4)

            results.append({
                "filename": filename,
                "subfolder": os.path.relpath(output_dir, folder_paths.get_output_directory()),
                "type": "output"
            })

            counter += 1

        print(f"[ImageSaverWithMetadata] Saved {len(results)} images to {output_dir}")
        return {"ui": {"images": results}}

    @staticmethod
    def _get_next_counter(output_dir, prefix):
        """Find the next available counter for filenames"""
        import os
        import re

        if not os.path.exists(output_dir):
            return 1

        existing_files = [f for f in os.listdir(output_dir) if f.startswith(prefix) and f.endswith('.png')]
        if not existing_files:
            return 1

        counters = []
        pattern = re.compile(rf"{re.escape(prefix)}_(\d+)\.png")
        for f in existing_files:
            match = pattern.match(f)
            if match:
                counters.append(int(match.group(1)))

        return max(counters) + 1 if counters else 1


NODE_CLASS_MAPPINGS = {
    "PromptLoraTestBench": PromptLoraTestBench,
    "ImageAnnotator": ImageAnnotator,
    "ImageSaverWithMetadata": ImageSaverWithMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptLoraTestBench": "Prompt & LoRA Test Bench",
    "ImageAnnotator": "Image Annotator",
    "ImageSaverWithMetadata": "Save Images with Metadata",
}
