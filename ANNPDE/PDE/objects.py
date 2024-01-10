from typing import List, Tuple, Type, Union, Literal
from sympy.parsing.sympy_parser import parse_expr
from .utilities import IntervalCondition
from .shapes import BaseShape
from numbers import Number
import numpy as np
import sympy as sp


class ReverseChauchyPDE:

    """
    Loss Function
    ------------------------------------------------------------------------------------

    This class is used to define a system of equations. It is used to define the 
    Linear Reverse Cauchy PDE problems that are to be solved by the neural networks.

    Parameters:

        f_function (str): f => u(X, t)                                       --->  (Γ×T)
        g_function (List[str]): g => ∂u(X)/∂n = [∂u(X)/∂x1, ∂u(X)/∂x2, ...]  --->  (Γ×T)
        h_function (str): h => u(X, 0)                                       --->   (Ω)

        ** Note that L(u(X, t)) is one of the parameters for calculating loss-function 
        ∂u(X, t)/∂t + L(u(X, t)) = 0

        ** Note that every symbol must be a string started with lowercase letter and 
        and other characters can be eighter lowercase letters or numbers or underscore 

        x_symbols (List[str]): ['x1', 'x2', ...] (time_symbol is not included)
        time_symbol (str): Any time symbol different from x_symbols

        domain_sample (np.ndarray): sample of domain
        edge_sample (np.ndarray): sample of edge
        t_condition (IntervalCondition): IntervalCondition object for time variable
    """

    def __init__(
            self,
            f_function: str,                # Γ × T -  f => u(X, t)
            g_function: List[str],          # Γ × T -  g => (∂u(X)/∂x1, ..., ∂u(X)/∂xn)
            h_function: str,                #   Ω   -  h => u(X, 0)
            x_symbols: List[str],           # ['x1', 'x2', ...] except t,
            domain_sample: np.ndarray,      # sample of domain
            edge_sample: np.ndarray,        # sample of edge
            time_sample: np.ndarray,           # sample of time
            time_symbol: str = 't'          # time symbol
        ) -> "EquationSystem":
        
        f_func, g_func, h_func, x_syms, t_sym, domain_sample, edge_sample, t_sample = \
            self.__error_handling(
                f_function, 
                g_function, 
                h_function, 
                x_symbols, 
                time_symbol,
                domain_sample,
                edge_sample,
                time_sample
            )
        
        self.x_dim = len(x_syms)
        self.x_vars = list(sp.symbols(list(map(self.__validate_symbol, x_syms))))
        self.t_var = sp.symbols(self.__validate_symbol(t_sym))
        
        self.f_func = parse_expr(f_func)
        self.g_func = [parse_expr(g) for g in g_func]
        self.h_func = parse_expr(h_func)

        self.edge_sample, self.domain_sample = edge_sample, domain_sample
        self.time_sample = t_sample

    @staticmethod
    def __error_handling(
            f_function: str,
            g_function: List[str],
            h_function: str,
            x_symbols: List[str],
            time_symbol: str,
            domain_sample: np.ndarray,
            edge_sample: np.ndarray,
            time_sample: np.ndarray
        ) -> Tuple[
            str, 
            List[str], 
            str, 
            List[str], 
            str
        ]:

        """
        This function is used to validate the input parameters of the EquationSystem
        
        --------------------------------------------------------------------------------
        
        Parameters:
            f_function (str): f => u(X, t)
            g_function (List[str]): g => ∂u(X)/∂n = [∂u(X)/∂x1, ∂u(X)/∂x2, ...]
            h_function (str): h => u(X, 0)
            x_symbols (List[str]): ['x1', 'x2', ...] (time_symbol is not included)
            domain_sample (np.ndarray): sample of domain
            edge_sample (np.ndarray): sample of edge
            t_condition (IntervalCondition): IntervalCondition object
            time_symbol (str): Any time symbol different from x_symbols
        Returns:
            f_function (str): f => u(X, t)
            g_function (List[str]): g => ∂u(X)/∂n = [∂u(X)/∂x1, ∂u(X)/∂x2, ...]
            h_function (str): h => u(X, 0)
            x_symbols (List[str]): ['x1', 'x2', ...] (time_symbol is not included)
            time_symbol (str): Any time symbol different from x_symbols
            domain_sample (np.ndarray): sample of domain
            edge_sample (np.ndarray): sample of edge
            t_condition (IntervalCondition): IntervalCondition object
        """

        # Validating functions
        if not isinstance(f_function, str):
            raise TypeError('f_function must be a string.')
        if not isinstance(g_function, (list, tuple)) or \
            not all(isinstance(g, str) for g in g_function):
            raise TypeError('g_function must be a list of strings.')
        if not isinstance(h_function, str):
            raise TypeError('h_function must be a string.')

        f_function = f_function.strip()
        g_function = [g.strip() for g in g_function]
        h_function = h_function.strip()

        # Validating x_symbols
        if not isinstance(x_symbols, (list, tuple)) or \
            not all(isinstance(x, str) for x in x_symbols):
            raise TypeError('x_symbols must be a list of strings.')
        x_symbols = [x.strip() for x in x_symbols]\
        
        if type(domain_sample) != np.ndarray:
            raise TypeError('domain_sample must be a numpy.ndarray.')
        if type(edge_sample) != np.ndarray:
            raise TypeError('edge_sample must be a numpy.ndarray.')
        if domain_sample.shape[1] != len(x_symbols):
            raise ValueError('domain_sample must have the same shape as x_symbols.')
        if edge_sample.shape[1] != len(x_symbols):
            raise ValueError('edge_sample must have the same shape as x_symbols.')

        # Validating time_symbol
        if not isinstance(time_symbol, str):
            raise TypeError('time_symbol must be a string.')
        time_symbol = time_symbol.strip()
        if time_symbol in x_symbols:
            raise ValueError('time_symbol must be different from x_symbols.')
        
        if type(time_sample) != np.ndarray:
            raise TypeError('time_sample must be a numpy.ndarray.')
        if len(time_sample.shape) != 1:
            raise ValueError('time_sample must be one dimensional.')
            
        return (
            f_function, 
            g_function, 
            h_function, 
            x_symbols, 
            time_symbol,
            domain_sample,
            edge_sample,
            time_sample
        )

    @staticmethod
    def __validate_symbol(symbol: str) -> str:
            
        """
        This function is used to validate a symbol
        
        --------------------------------------------------------------------------------
        
        Parameters:
            symbol (str): symbol must be a string
        Returns:
            symbol (str): symbol must be a string
        """
    
        if not symbol[0].islower():
            raise ValueError('symbol must start with a lowercase letter.')
        if not all(c.islower() or c.isdigit() or c == '_' for c in symbol):
            raise ValueError(
                'symbol must contain only lowercase letters, numbers and underscore.'
            )
        return symbol


    def f_function(
            self,
        ):
        pass