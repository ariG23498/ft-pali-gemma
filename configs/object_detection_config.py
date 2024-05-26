import torch

DATASET_ID = "ariG23498/license-detection-paligemma"
MODEL_ID = "google/paligemma-3b-pt-224"
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
MODEL_DTYPE = torch.bfloat16
MODEL_REVISION = "bfloat16"
EPOCHS = 1
