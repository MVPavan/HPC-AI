from pathlib import Path
from typing import Optional
from pydantic import (
    BaseModel,
    ValidationError,
    ValidationInfo,
    field_validator,
    dataclasses
)
import math
import inspect
from enum import Enum

import torch
import torch.nn as nn
from torch.functional import F
from torchinfo import summary

torch.manual_seed(1337)