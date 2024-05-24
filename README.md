# FineTuning PaliGemma

## What is PaliGemma?

PaliGemma is a new family of vision-language models from Google. These models can process both images and text to produce text outputs.

Google has released three types of PaliGemma models:

Pretrained (pt) models: Trained on large datasets without task-specific tuning.

Mix models: A combination of pre-trained and fine-tuned elements.

Fine-tuned (ft) models: Optimized for specific tasks with additional training.

Each type comes in different resolutions and multiple precisions for convenience. All models are available on the Hugging Face Hub with model cards, licenses, and integration with transformers.


## Fine-Tuning Methods

1. JAX Fine-Tuning Script:

2. Fine-tuning using HuggingFace transformers

3. Fine-tuning using Vanilla Pytorch script (shown here)

## Results

Here, we fine-tune a custom Tuxemon dataset from HuggingFace Hub.

| Image 1 | Image 2 |
|---|---|
| ![image](https://github.com/ariG23498/ft-pali-gemma/assets/44690292/67d47985-ec4a-4e3f-ac45-d2474cf988d4) | ![image](https://github.com/ariG23498/ft-pali-gemma/assets/44690292/81c7ed90-9377-49e3-ad2c-4d680d350b67) |


## Citation

```
@misc{github_repository,
  author = {Aritra Roy Gosthipaty, Ritwik Raha}, 
  title = {ft-pali-gemma}, 
  publisher = {{GitHub}(https://github.com)},
  howpublished = {\url{https://github.com/ariG23498/ft-pali-gemma/edit/main/README.md}},
  year = {2024}  
}
```
