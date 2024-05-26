import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from datasets import load_dataset
from configs import vanilla_config

from paligemma_ft.data_utis import collate_fn
from paligemma_ft.model_utils import freeze_layers
from functools import partial
from matplotlib import pyplot as plt


def infer_on_model(model, test_batch):
    # hardcoding the index to get same before and after results
    index = 0

    # help from : https://discuss.huggingface.co/t/vitimageprocessor-output-visualization/76335/6
    mean = processor.image_processor.image_mean
    std = processor.image_processor.image_std

    pixel_value = test_batch["pixel_values"][index].cpu().to(torch.float32)

    unnormalized_image = (
        pixel_value.numpy() * np.array(std)[:, None, None]
    ) + np.array(mean)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)

    plt.imshow(
        unnormalized_image,
    )
    plt.axis("off")
    with torch.no_grad():
        generated_outputs = model.generate(
            **test_batch,
            max_new_tokens=100,
        )
        generated_outputs = processor.batch_decode(
            generated_outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[index]

    plt.figtext(
        0.5,
        0.01,
        generated_outputs,
        wrap=True,
        horizontalalignment="center",
        fontsize=12,
    )

    plt.show()


if __name__ == "__main__":
    # get the device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # load the dataset
    print(f"[INFO] loading {vanilla_config.DATASET_ID} from hub...")
    dataset = load_dataset(vanilla_config.DATASET_ID, split="train")

    # split into train and test
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    print(f"[INFO] {len(train_dataset)=}")
    print(f"[INFO] {len(test_dataset)=}")

    # get the processor
    print(f"[INFO] loading {vanilla_config.MODEL_ID} processor from hub...")
    processor = AutoProcessor.from_pretrained(vanilla_config.MODEL_ID)

    # build the data loaders
    print("[INFO] building the data loaders...")
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=partial(
            collate_fn,
            image_title="image",
            prompt="Caption the image.",
            suffix_title="blip_caption",
            processor=processor,
            device=device,
            train=True,
        ),
        batch_size=vanilla_config.BATCH_SIZE,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=partial(
            collate_fn,
            image_title="image",
            prompt="Caption the image.",
            suffix_title=None,
            processor=processor,
            device=device,
            train=False,
        ),
        batch_size=vanilla_config.BATCH_SIZE,
        shuffle=False,
    )

    # load the pre trained model
    print(f"[INFO] loading {vanilla_config.MODEL_ID} model...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        vanilla_config.MODEL_ID,
        torch_dtype=vanilla_config.MODEL_DTYPE,
        device_map=device,
        revision=vanilla_config.MODEL_REVISION,
    )

    # freeze the weights
    print(f"[INFO] freezing the model weights...")
    model = freeze_layers(model, not_to_freeze="attn")

    # run model generation before fine tuning
    test_batch = next(iter(test_dataloader))
    infer_on_model(model, test_batch)

    # fine tune the model
    print("[INFO] fine tuning the model...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=vanilla_config.LEARNING_RATE)
    for epoch in range(vanilla_config.EPOCHS):
        for idx, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            if idx % 10 == 0:
                print(f"Epoch: {epoch} Iter: {idx} Loss: {loss.item():.4f}")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # run model generation after fine tuning
    infer_on_model(model, test_batch)
