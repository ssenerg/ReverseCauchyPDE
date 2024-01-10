from ANNPDE.shapes import ElipseShape
from ANNPDE.shapes import CircleShape
from ANNPDE import ReverseChauchyPDE
from ANNPDE import IntervalCondition
from random import randint
import numpy as np


SEED = randint(1, 1000000)

TR_EXPR = 'E ** x1 * sin(x2) * cos(t)' # ??
F_EXPR = 'E ** x1 * sin(x2) * cos(t)'
G_EXPR = ['E ** x1 * sin(x2)', 'E ** x1 * cos(x2)']
H_EXPR = 'E ** x1 * sin(x2)'
Xs_symbol, t_symbol = ['x1', 'x2'], 't'

IntervalCondition.precision = 5

T_CONDITION = IntervalCondition('[0, pi/2]')

shape = CircleShape(
    seed=seed,
    center=np.array([0, 0]),
    radius=1
)

edge_sample, domain_sample = shape.get()

pde = ReverseChauchyPDE(
    tr_function=TR_EXPR,
    f_function=F_EXPR,
    g_function=G_EXPR,
    h_function=H_EXPR,
    x_symbols=Xs_symbol,
    shape=shape,
    t_condition=T_CONDITION,
    time_symbol=t_symbol
)

# TODO
neural_network = pde.get_neural_network()
