import torch
import os
from dotenv import load_dotenv

load_dotenv()

HF_HOME = os.getenv("HF_HOME", os.path.expanduser(".cache/huggingface"))
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
HF_CACHE = os.path.join(PROJECT_ROOT, HF_HOME)
