# ComfyUI Prompt & LoRA Test Bench

Extension pour ComfyUI permettant de tester automatiquement plusieurs prompts avec plusieurs LoRAs.

## Installation

Placez ce dossier dans `ComfyUI/custom_nodes/`

## Utilisation

### Node: Prompt & LoRA Test Bench

Ce node génère automatiquement toutes les combinaisons de prompts et LoRAs.

**Entrées:**
- `model`: Le modèle de base
- `clip`: Le modèle CLIP
- `prompts`: Liste de prompts (un par ligne)
- `lora_names`: Liste de noms de fichiers LoRA (un par ligne, ex: `my_lora.safetensors`)
- `lora_strength_model`: Force du LoRA sur le modèle (défaut: 1.0)
- `lora_strength_clip`: Force du LoRA sur CLIP (défaut: 1.0)
- `negative_prompt` (optionnel): Prompt négatif commun à toutes les générations

**Sorties (listes):**
- `model`: Modèles avec LoRA appliqué
- `clip`: CLIP avec LoRA appliqué
- `positive`: Conditionnements positifs
- `negative`: Conditionnements négatifs
- `prompt_text`: Textes des prompts
- `lora_name`: Noms des LoRAs

### Node: Text Combiner

Utilitaire pour créer des noms de fichiers à partir du prompt et du nom du LoRA.

**Entrées:**
- `prompt`: Texte du prompt
- `lora_name`: Nom du LoRA
- `separator`: Séparateur (défaut: "_")

**Sortie:**
- `STRING`: Texte combiné et nettoyé pour nom de fichier

## Exemple de workflow

1. Connectez votre `Load Checkpoint` → `Prompt & LoRA Test Bench`
2. Entrez vos prompts (un par ligne)
3. Entrez vos noms de LoRA (un par ligne)
4. Connectez les sorties `positive` et `negative` au `KSampler`
5. Connectez `model` au `KSampler`
6. Utilisez `Text Combiner` avec `prompt_text` et `lora_name` pour générer des noms de fichiers uniques
7. Le workflow s'exécutera automatiquement pour toutes les combinaisons

## Fonctionnement

Si vous avez:
- 3 prompts
- 2 LoRAs

Le node générera **6 combinaisons** (3 × 2) et les traitera automatiquement en batch.

## Notes

- Les fichiers LoRA doivent être dans votre dossier `ComfyUI/models/loras/`
- Les noms de fichiers doivent inclure l'extension (`.safetensors` ou `.ckpt`)
- Les générations s'enchaînent automatiquement sans interaction
