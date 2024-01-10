from sympy.parsing.sympy_parser import parse_expr
from numbers import Number


class IntervalCondition:

    """
    Interval Condition
    ------------------------------------------------------------------------------------

    This class is used to define a interval condition. It is used to define the 
    boundary conditions of the PDE equations that are to be solved by the neural 
    networks.

    Parameters:
        condition (str): condition must be in format [0, pi/2] or [0, e) or (0, 1) or 
        (0, 1)
    Attributes:
        left_value (str): Left value of the interval
        right_value (str): Right value of the interval
        left_closed (bool): True if the left value is included in the interval
        right_closed (bool): True if the right value is included in the interval
        
    """

    precision = 15
    
    def __init__(
            self,
            condition: str, # in format [0, pi/2] or [0, e) or (0, 1) or (0, 1],
        ) -> "IntervalCondition":

        if not isinstance(condition, str):
            raise TypeError('condition must be a string.')

        consition = condition.strip()
        if condition[0] not in ['[', '('] or condition[-1] not in [']', ')']:
            raise ValueError(
                'condition must be in format [0, pi/2] or [0, E) or (0, 1) or (0, 1]'
            )
        if condition[0] == '[':
            self.left_closed = True
        else:
            self.left_closed = False
        if condition[-1] == ']':
            self.right_closed = True
        else:
            self.right_closed = False
        
        condition = condition[1:-1].replace(' ', '')
        condition = condition.split(',')
        if len(condition) != 2:
            raise ValueError(
                'condition must be in format [0, pi/2] or [0, E) or (0, 1) or (0, 1]'
            )

        try:
            self.left_value = parse_expr(condition[0].strip()).evalf(self.precision)
        except TypeError:
            raise ValueError(
                'condition must be in format [0, pi/2] or [0, E) or (0, 1) or (0, 1]'
            )

        try:
            self.right_value = parse_expr(condition[1].strip()).evalf(self.precision)
        except TypeError:
            raise ValueError(
                'condition must be in format [0, pi/2] or [0, E) or (0, 1) or (0, 1]'
            )

    def __str__(self) -> str:

        string = ''

        if self.left_closed:
            string += '['
        else:
            string += '('

        if self.right_closed:
            string += f'{self.left_value}, {self.right_value}]'
        else:
            string += f'{self.left_value}, {self.right_value})'

        return string

    def is_in(self, value: Number) -> bool:
        
        """
        This function is used to check if a value is in the interval
        
        --------------------------------------------------------------------------------
        
        Parameters:
            value (Number): value to be checked
        Returns:
            bool: True if value is in the interval
        """

        if self.left_closed:
            if value < self.left_value:
                return False
        else:
            if value <= self.left_value:
                return False

        if self.right_closed:
            if value > self.right_value:
                return False
        else:
            if value >= self.right_value:
                return False

        return True 


def int_validator(number: Number) -> bool:

    if not isinstance(number, Number):
        raise TypeError('number must be a number.')
    try:
        int(number)
    except ValueError:
        return False
    return True
