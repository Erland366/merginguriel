import torch
import os
from dotenv import load_dotenv
from loguru import logger
from merginguriel.utils import seed_everything


load_dotenv()

global HF_HOME
HF_HOME = os.getenv("HF_HOME", os.path.expanduser(".cache/huggingface"))

global PROJECT_ROOT
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))

global HF_CACHE
HF_CACHE = os.path.join(PROJECT_ROOT, HF_HOME)

seed_everything(3147)
