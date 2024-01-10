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
    PDE equations that are to be solved by the neural networks.

    Parameters:
        f_function (str): f => u(X, t)
        g_function (List[str]): g => ∂u(X)/∂n = [∂u(X)/∂x1, ∂u(X)/∂x2, ...]
        h_function (str): h => u(X, 0)

        ** Note that every symbol must be a string started with lowercase letter and 
        and other characters can be eighter lowercase letters or numbers or underscore 

        x_symbols (List[str]): ['x1', 'x2', ...] (time_symbol is not included)
        conditions (List[IntervalCondition]): List of IntervalCondition objects
        time_symbol (str): Any time symbol different from x_symbols

    Attributes:
        x_dim (int): Dimension of the x_symbols
        x_vars (List[sympy.Symbol]): List of x_symbols
        t_var (sympy.Symbol): Time symbol
        conditions (List[IntervalCondition]): List of IntervalCondition objects
        f_func (sympy.Expr): f => u(X, t)
        g_func (List[sympy.Expr]): g => ∂u(X)/∂n = [∂u(X)/∂x1, ∂u(X)/∂x2, ...]
        h_func (sympy.Expr): h => u(X, 0)
    """

    def __init__(
            self,
            tr_function: str,               # tr => ∂u(X, t)/∂t + L(u(X, t)) = 0
            f_function: str,                # f => u(X, t)
            g_function: List[str],          # g => ∂u(X)/∂n = [∂u(X)/∂x1, ∂u(X)/∂x2, ...]
            h_function: str,                # h => u(X, 0)
            x_symbols: List[str],           # ['x1', 'x2', ...] except t,
            shape: Type[BaseShape],
            t_condition: IntervalCondition, # t ∈ [0, 1]
            time_symbol: str = 't'
        ) -> "EquationSystem":
        
        tr_func, f_func, g_func, h_func, x_syms, shape, t_cond, t_sym = \
            self.__error_handling(
                tr_function,
                f_function, 
                g_function, 
                h_function, 
                x_symbols, 
                shape, 
                t_condition,
                time_symbol
            )
        
        self.x_dim = len(x_syms)
        self.x_vars = list(sp.symbols(list(map(self.__validate_symbol, x_syms))))
        self.t_var = sp.symbols(self.__validate_symbol(t_sym))
        
        self.tr_func = parse_expr(tr_func)
        self.f_func = parse_expr(f_func)
        self.g_func = [parse_expr(g) for g in g_func]
        self.h_func = parse_expr(h_func)

        self.t_cond = t_cond
        self.edge_sample, self.domain_sample = shape.get()

    @staticmethod
    def __error_handling(
            tr_function: str,
            f_function: str,
            g_function: List[str],
            h_function: str,
            x_symbols: List[str],
            shape: Type[BaseShape],
            t_condition: IntervalCondition,
            time_symbol: str
        ) -> Tuple[
            str, 
            List[str], 
            str, 
            List[str], 
            Type[BaseShape], 
            IntervalCondition, 
            str
        ]:

        """
        This function is used to validate the input parameters of the EquationSystem
        
        --------------------------------------------------------------------------------
        
        Parameters:
            tr_function (str): tr => ∂u(X, t)/∂t + L(u(X, t)) = 0
            f_function (str): f => u(X, t)
            g_function (List[str]): g => ∂u(X)/∂n = [∂u(X)/∂x1, ∂u(X)/∂x2, ...]
            h_function (str): h => u(X, 0)
            x_symbols (List[str]): ['x1', 'x2', ...] (time_symbol is not included)
            shape (Type[BaseShape]): Shape object
            t_condition (IntervalCondition): IntervalCondition object
            time_symbol (str): Any time symbol different from x_symbols
        Returns:
            tr_function (str): tr => ∂u(X, t)/∂t + L(u(X, t)) = 0
            f_function (str): f => u(X, t)
            g_function (List[str]): g => ∂u(X)/∂n = [∂u(X)/∂x1, ∂u(X)/∂x2, ...]
            h_function (str): h => u(X, 0)
            x_symbols (List[str]): ['x1', 'x2', ...] (time_symbol is not included)
            shape (Type[BaseShape]): Shape object
            t_condition (IntervalCondition): IntervalCondition object
            time_symbol (str): Any time symbol different from x_symbols
        """

        # Validating functions
        if not isinstance(tr_function, str):
            raise TypeError('tr_function must be a string.')
        if not isinstance(f_function, str):
            raise TypeError('f_function must be a string.')
        if not isinstance(g_function, (list, tuple)) or \
            not all(isinstance(g, str) for g in g_function):
            raise TypeError('g_function must be a list of strings.')
        if not isinstance(h_function, str):
            raise TypeError('h_function must be a string.')

        tr_function = tr_function.strip()               
        f_function = f_function.strip()
        g_function = [g.strip() for g in g_function]
        h_function = h_function.strip()

        # Validating x_symbols
        if not isinstance(x_symbols, (list, tuple)) or \
            not all(isinstance(x, str) for x in x_symbols):
            raise TypeError('x_symbols must be a list of strings.')
        x_symbols = [x.strip() for x in x_symbols]

        # Validating time_symbol
        if not isinstance(time_symbol, str):
            raise TypeError('time_symbol must be a string.')
        time_symbol = time_symbol.strip()
        if time_symbol in x_symbols:
            raise ValueError('time_symbol must be different from x_symbols.')
        
        # Validating t_condition
        if not isinstance(t_condition, IntervalCondition):
            raise TypeError('t_condition must be an IntervalCondition object.')

        return (
            tr_function,
            f_function, 
            g_function, 
            h_function, 
            x_symbols, 
            shape,
            t_condition, 
            time_symbol
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
    
    def __str__(self) -> str:
        string = 'Equation System:\n'
        string += f'  ⚬ u(X, t) = {self.f_func}\n'
        string += f'  ⚬ ∂u(X)/∂n = {self.g_func}\n'
        string += f'  ⚬ h(X, t) = {self.h_func}\n'
        string += f'  ⚬ X = {tuple(self.x_vars)}\n'
        string += f'  ⚬ time variable = {self.t_var}\n'
        string += '  ⚬ conditions:\n'
        for i, c in enumerate(self.x_conds):
            string += f'    - {self.x_vars[i]} ∈ {c}\n'
        string += f'\n    - {self.t_var} ∈ {self.t_cond}\n'
        return string

    def loss_function(
            self, 
            x: np.ndarray, 
            t: Number, 
            value: Number
        ) -> Number:
        
        """
        This function is used to get the loss function of the Neural Network

        --------------------------------------------------------------------------------

        Parameters:
            x (np.ndarray): x ∈ R^n
            t (Number): time
            value (Number): value of the u function at x and time t
        Returns:
            loss (Number): loss function of the Neural Network
        """

        # J(u) = 
        #        ||∂u(X, t)/∂t + L(u(X, t))|| + 
        #        ||∂u(X)/∂n - g(X)||² + 
        #        ||u(X, t) - f(X, t)||² + 
        #        ||u(X, 0) - h(X, 0)||²
        


    def initial_layer(self) -> np.ndarray: # EDGE, DOMAIN

        """
        TODO
        """

        t = 0
        function = 'h' # TODO: ?

        return self.__call__(func, t)[0]
    
    # TODO: validate function and t for the actual domain of calling

    def __call__(
            self,
            func: Literal['f', 'g', 'h'], 
            t: Number,
            d: Literal['domain', 'edge'] = None,
        ) -> Union[Number, np.ndarray]:

        """
        This function is used to get the value of the function func at time t

        --------------------------------------------------------------------------------

        Parameters:
            func (Literal['h', 'f', 'g']): function to be called
            t (Number): time
            d (Literal['domain', 'edge']): if None, the function is called for the whole 
            shape, if 'domain', the function is called for the domain of the shape, if 
            'edge', the function is called for the edge of the shape
        Returns:
            value (Union[Number, np.ndarray]): value of the function func at time t
        """

        if d not in ['domain', 'edge', None]:
            raise ValueError('d must be either "domain" or "edge" or None.')
        if not isinstance(t, Number):
            raise TypeError('t must be a number.')
        if not self.t_cond.is_in(t):
            raise ValueError(f't must be in {self.t_cond}.')
        if func == '':
            pass
