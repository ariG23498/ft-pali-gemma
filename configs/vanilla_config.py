import torch

DATASET_ID = "diffusers/tuxemon"
MODEL_ID = "google/paligemma-3b-pt-224"
BATCH_SIZE = 4
MODEL_DTYPE = torch.bfloat16
MODEL_REVISION = "bfloat16"
LEARNING_RATE = 5e-5
EPOCHS = 5
