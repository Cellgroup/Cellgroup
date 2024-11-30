" Functions for reading and writing data."
import os
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import tifffile as tiff
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm