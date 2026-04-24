import os
import torch
from dotenv import load_dotenv
from merginguriel.utils import seed_everything

try:
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover - fallback if loguru missing
    import logging

    logger = logging.getLogger("merginguriel")


load_dotenv()

global HF_HOME
HF_HOME = os.getenv("HF_HOME", os.path.expanduser(".cache/huggingface"))

global PROJECT_ROOT
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))

global HF_CACHE
HF_CACHE = os.path.join(PROJECT_ROOT, HF_HOME)

seed_everything(3147)
