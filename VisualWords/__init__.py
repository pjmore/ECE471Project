from __future__ import annotations
from typing import *
import numpy as np #type: ignore

from .core import ImageFeatures
from .Patches import MakeGrayPatchExtractor, MakeColourPatchExtractor
from .SIFT import MakeGraySIFTExtractor, MakeColourSIFTExtractor, MakeSparseGraySIFTExtractor


