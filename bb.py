from ANNPDE.PDE import ReverseChauchyPDE
from ANNPDE.PDE.shapes import (
    ElipseShape, 
    CircleShape, 
    LineShape
)
from ANNPDE.ANN import (
    LSTM, 
    laplacian ,
    derivative,
    prepare_data
)
import plotly.graph_objs as go
from random import randint
from torch import nn
import numpy as np
import torch


SEED = randint(1, 1000000)
print('Seed:', SEED)