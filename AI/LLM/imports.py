from pathlib import Path

from pydantic import (
    BaseModel,
    ValidationError,
    ValidationInfo,
    field_validator,
)

import torch
import torch.nn as nn
from torch.functional import F

torch.manual_seed(1337)
