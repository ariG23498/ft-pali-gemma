import torch


def collate_fn(examples, image_title, prompt, suffix_title, processor, device, train):
    images = [example[image_title].convert("RGB") for example in examples]

    prompt = [prompt for _ in examples]
    if train:
        suffix = [example[suffix_title] for example in examples]
    else:
        suffix = None

    # Help from: https://github.com/huggingface/transformers/issues/30987
    inputs = processor(
        images=images,
        text=prompt,
        suffix=suffix,
        return_tensors="pt",
        padding="longest",
    )

    inputs = inputs.to(torch.bfloat16).to(device)
    return inputs
