from sympy.parsing.sympy_parser import parse_expr
from numbers import Number


def int_validator(number: Number) -> bool:

    if not isinstance(number, Number):
        raise TypeError('number must be a number.')
    try:
        int(number)
    except ValueError:
        return False
    return True



