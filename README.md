# ComfyUI Prompt & LoRA Test Bench

Extension for ComfyUI to automatically test multiple prompts with multiple LoRAs.

## Installation

Place this folder in `ComfyUI/custom_nodes/`

Demo workflow: resources/demo.json

## Nodes

### 1. Prompt & LoRA Test Bench

This node automatically generates all combinations of prompts and LoRAs for batch testing.

**How it works:**
- Takes N prompts and M LoRAs as input
- Generates N × M combinations automatically
- Processes all combinations in a single workflow run

**Inputs:**
- `model`: Base model
- `clip`: CLIP model
- `prompts`: List of prompts (one per line)
- `negative_prompts`: List of negative prompts (one per line, optional)
- `lora_names`: List of LoRA names (one per line, **extensions optional**)
- `lora_directory`: Directory to scan for LoRAs, or use "(use lora_names field)" to manually list
- `lora_strength_model`: LoRA strength for the model
- `lora_strength_clip`: LoRA strength for CLIP
- `prompt` (optional input): Single prompt override
- `negative_prompt` (optional input): Single negative prompt override

**Outputs (lists):**
- `model`: Models with LoRA applied
- `clip`: CLIP with LoRA applied
- `positive`: Positive conditionings
- `negative`: Negative conditionings
- `lora_names`: LoRA names used
- `prompts`: Prompt texts used
- `negative_prompts`: Negative prompt texts used

**Example:**
With 3 prompts and 2 LoRAs, this node generates **6 images** automatically processed in batch. Each prompt processed for one LoRA, then proceeds to the next LoRA, and so on.

**Special features:**
- Can use a whole directory and filter a range
- Supports `<lora:name:strength>` format for per-LoRA strength control (e.g., `<lora:style_lora:0.8>`)
- Auto-detects file extensions (`.safetensors`, `.ckpt`, `.pt`, `.bin`)
- Searches subdirectories recursively

### 2. Image Annotator

Adds text annotations to images with LoRA names burned into the image.

**Inputs:**
- `images`: Images to annotate (list)
- `lora_names`: LoRA names to display (list, force input)
- `enable_annotations`: Enable/disable annotations (default: True)
- `preserve_original`: Keep both original and annotated versions (default: False)
- `font_size`: Text size (default: 18, range: 8-128)

**Output:**
- `images`: Annotated images (and originals if preserve_original=True)

**How it works:**
- Draws LoRA name in bottom-left corner with black background
- Uses Arial or DejaVuSans font (falls back to default)
- If preserve_original is True, outputs 2× images (original + annotated)

### 3. Save Images with Metadata

Enhanced image saving node that embeds prompt and LoRA information in PNG metadata.

**Inputs:**
- `images`: Images to save (list)
- `filename_prefix`: Prefix for filenames (default: "ComfyUI", supports paths)
- `prompts` (optional): Prompt texts to embed (list, force input)
- `negative_prompts` (optional): Negative prompt texts to embed (list, force input)
- `lora_names` (optional): LoRA names to embed (list, force input)

**Features:**
- Saves PNG images with sequential numbering (`prefix_00001.png`, `prefix_00002.png`, ...)
- Embeds individual metadata fields: `prompt`, `negative_prompt`, `lora`
- Embeds combined JSON in `parameters` field for easier parsing
- Supports subfolder creation via prefix (e.g., `subfolder/prefix`)
- Auto-increments counter based on existing files
- Compatible with standard PNG readers

## Workflow Example

1. Connect `Load Checkpoint` → `Prompt & LoRA Test Bench`
2. Enter your prompts (one per line) in the test bench node
3. Enter your LoRA names (one per line, **extensions optional**)
4. Connect test bench outputs:
   - `positive` and `negative` → `KSampler`
   - `model` → `KSampler`
5. Connect `KSampler` → `Image Annotator` (optional, to burn LoRA names into images)
6. Connect `lora_names` output → `Image Annotator`
7. Connect images → `Save Images with Metadata`
8. Connect `prompts`, `negative_prompts`, and `lora_names` → save node for metadata
9. Run workflow once - all combinations process automatically

## Naming Constraints & Possibilities

**LoRA naming (in `lora_names` field):**
- **Extensions are OPTIONAL** - node auto-detects `.safetensors`, `.ckpt`, `.pt`, `.bin`
- Examples:
  - `my_lora` → searches for `my_lora.safetensors`, `my_lora.ckpt`, etc.
  - `my_lora.safetensors` → uses exact filename
  - `subfolder/my_lora` → searches in subdirectory (extension optional)
  - `<lora:my_lora:0.8>` → uses LoRA with custom strength 0.8
- Searches recursively through all subdirectories in `ComfyUI/models/loras/`
- Case-sensitive on Linux/Mac, case-insensitive on Windows

**Negative prompt handling:**
- Can provide one negative prompt per line (matched to prompts)
- If fewer negatives than prompts, last negative is repeated
- If empty, uses empty string for all negatives

**LoRA directory mode:**
- Set `lora_directory` to a path to auto-scan that folder
- All `.safetensors`, `.ckpt`, `.pt`, `.bin` files are loaded automatically
- Overrides `lora_names` field when directory is selected

**Special features:**
- Works without LoRAs: outputs combinations with `no_lora` label
- Supports per-LoRA strength: `<lora:name:0.5>` format
- Negative strength values supported (range: -20.0 to 20.0)

## Examples

**Basic usage:**
```
prompts:
a cat
a dog

lora_names:
anime_style
realistic_v2
```
Result: 4 images (2 prompts × 2 LoRAs)

**With custom strength:**
```
prompts:
portrait of a woman

lora_names:
<lora:style_lora:0.8>
<lora:detail_lora:1.5>
```
Result: 2 images with different LoRA strengths

**Using directory mode:**
```
lora_directory: E:\ComfyUI\models\loras\characters
```
Result: All LoRAs in that folder × all prompts

## Notes

- LoRA files must be in `ComfyUI/models/loras/` or subdirectories
- Extensions `.safetensors`, `.ckpt`, `.pt`, `.bin` supported
- All combinations process automatically in batch mode
- Use metadata-embedded images for easier tracking
- Image Annotator useful for visual identification of which LoRA was used
