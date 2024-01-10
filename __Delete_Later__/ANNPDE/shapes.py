from numpy import random as numpy_random
from .utilities import int_validator
from abc import ABC, abstractmethod
import plotly.graph_objects as go
import random as python_random
from numbers import Number
from typing import Tuple
import numpy as np


class BaseShape(ABC):

    """
    Base Shape
    ------------------------------------------------------------------------------------
    
    This class is used to define a shape. It is used to define the boundary conditions
    of the PDE equations that are to be solved by the neural networks.
    Also it is used to generate sample points on the edge and in the domain of the
    shape and visualize the shape
    
    Parameters:
        seed (int): seed for random number generator
        edge_n (int): number of sample points on the edge of the shape
        domain_n (int): number of sample points in the domain of the shape
        cross_sample_generate (Number): number of sample points generated for each 
        sample point on the edge of the shape
    Attributes:
        edge_n (int): number of sample points on the edge of the shape
        domain_n (int): number of sample points in the domain of the shape
        cross_sample (Number): number of sample points generated for each 
        sample point on the edge of the shape
        even_sample (bool): if True, the sample points are generated evenly
        dim (int): dimension of the shape
        edge_sample (np.ndarray): sample points on the edge of the shape
        domain_sample (np.ndarray): sample points in the domain of the shape
    """
    
    def __init__(
            self, 
            seed: int,
            edge_n: int,
            domain_n: int,
            cross_sample_generate: Number,
        ) -> None:
        
        if not int_validator(seed):
            raise TypeError('seed must be an integer.')
        elif not 0 <= seed <= 2 ** 32 - 1:
            raise ValueError('seed must be in range [0, 2**32 - 1].')

        if not int_validator(edge_n):
            raise TypeError('edge_n must be an integer.')
        elif edge_n <= 0:
            raise ValueError('edge_n must be positive.')
        elif not int_validator(domain_n):
            raise TypeError('domain_n must be an integer.')
        elif domain_n <= 0:
            raise ValueError('domain_n must be positive.')
        elif not isinstance(cross_sample_generate, Number):
            raise TypeError('cross_sample_generate must be a number.')
        elif cross_sample_generate <= 1:
            raise ValueError('cross_sample_generate must be greater than 1.')
        
        numpy_random.seed(seed)
        python_random.seed(seed)

        self.edge_n = edge_n
        self.domain_n = domain_n
        self.cross_sample = cross_sample_generate

        self.edge_sample = None
        self.domain_sample = None

    def _edge_sample(self) -> np.ndarray:

        """
        This function is used to get the sample points on the edge of the shape

        --------------------------------------------------------------------------------

        Returns:
            edge_sample (np.ndarray): sample points on the edge of the shape
        """
        
        self.edge_sample = np.array([])
        self._instant_edge_sample()
    
    def _domain_sample(self) -> np.ndarray:

        """
        This function is used to get the sample points in the domain of the shape

        --------------------------------------------------------------------------------

        Returns:
            domain_sample (np.ndarray): sample points in the domain of the shape
        """
        
        self.domain_sample = np.array([])
        self._instant_domain_sample()

    @abstractmethod
    def _dim(self) -> int:

        """
        This function is used to get the dimension of the shape
        """
        pass
    
    @abstractmethod
    def _instant_domain_sample(self) -> None:
        pass

    @abstractmethod
    def _instant_edge_sample(self) -> None:
        pass

    def get(self) -> Tuple[np.ndarray, np.ndarray]:

        """
        This function is used to get the sample points of the shape

        --------------------------------------------------------------------------------

        Returns:
            edge_sample (np.ndarray): sample points on the edge of the shape
            domain_sample (np.ndarray): sample points in the domain of the shape
        """

        self._edge_sample()
        self._domain_sample()

        return self.edge_sample, self.domain_sample

    def plot(
            self,
            edge_values: np.ndarray,
            domain_values: np.ndarray
        ) -> None:
            
        """
        This function is used to plot the shape

        --------------------------------------------------------------------------------

        Parameters:
            edge_values (np.ndarray): values of the edge sample points
            domain_values (np.ndarray): values of the domain sample points
        """

        if not isinstance(edge_values, np.ndarray):
            raise TypeError('edge_values must be np.ndarray.')
        if not isinstance(domain_values, np.ndarray):
            raise TypeError('domain_values must be np.ndarray.')
        if edge_values.shape != (self.edge_n, ):
            raise ValueError('edge_values must have shape (edge_n, ).')
        if domain_values.shape != (self.domain_n, ):
            raise ValueError('domain_values must have shape (domain_n, ).')
        
        edge_x = self.edge_sample[:, 0]
        edge_y = self.edge_sample[:, 1]
        domain_x = self.domain_sample[:, 0]
        domain_y = self.domain_sample[:, 1]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='markers',
            marker=dict(
                color=edge_values,
                colorscale='Viridis',
                line_width=1
            )
        ))
        fig.add_trace(go.Scatter(
            x=domain_x,
            y=domain_y,
            mode='markers',
            marker=dict(
                color=domain_values,
                colorscale='Viridis',
                line_width=1
            )
        ))
        fig.show()


class ElipseShape(BaseShape):

    """
    Elipse Shape
    ------------------------------------------------------------------------------------
    
    This class is used to define a elipse shape. It is used to define the boundary
    conditions of the PDE equations that are to be solved by the neural networks.
    Also it is used to generate sample points on the edge and in the domain of the
    elipse and visualize the elipse
    
    Parameters:
        seed (int): seed for random number generator
        center (np.ndarray): center of the elipse
        v_radius (Number): vertical radius of the elipse
        h_radius (Number): horizontal radius of the elipse
    """
    
    def __init__(
            self, 
            seed: int,
            edge_n: int,
            domain_n: int,
            cross_sample_generate: Number,
            center: np.ndarray, 
            v_radius: Number, 
            h_radius: Number,
        ) -> None:

        super().__init__(
            seed,
            edge_n,
            domain_n,
            cross_sample_generate,
        )

        if not isinstance(center, np.ndarray):
            if not isinstance(center, (list, tuple)):
                raise TypeError('center must be np.ndarray or list or tuple.')
            center = np.array(center)
            
        if not isinstance(v_radius, Number):
            raise TypeError('v_radius must be a number.')
        if not isinstance(h_radius, Number):
            raise TypeError('h_radius must be a number.')
        if v_radius <= 0:
            raise ValueError('v_radius must be positive.')
        if h_radius <= 0:
            raise ValueError('h_radius must be positive.')

        self.center = center
        self.v_radius = v_radius
        self.h_radius = h_radius

        self.dim = self._dim()

    def _dim(self) -> int:

        """
        This function is used to get the dimension of the Elipse / Circle
        """

        length = len(self.center)
        if length != 2:
            raise ValueError('center must be a 2D vector.')
        return length
    
    def _instant_domain_sample(self) -> None:

        theta = np.random.uniform(0, 2 * np.pi, self.domain_n)
        h_radius = np.random.uniform(0, self.h_radius, self.domain_n)
        v_radius = np.random.uniform(0, self.v_radius, self.domain_n)

        x = self.center[0] + h_radius * np.cos(theta)
        y = self.center[1] + v_radius * np.sin(theta)

        self.domain_sample = np.column_stack((x, y))

    def _instant_edge_sample(self) -> None:
            
        theta = np.random.uniform(0, 2 * np.pi, self.edge_n)
        h_radius = np.full(self.edge_n, self.h_radius)
        v_radius = np.full(self.edge_n, self.v_radius)

        x = self.center[0] + h_radius * np.cos(theta)
        y = self.center[1] + v_radius * np.sin(theta)

        self.edge_sample = np.column_stack((x, y))


class CircleShape(ElipseShape):

    """
    Circle Shape
    ------------------------------------------------------------------------------------

    This class is used to define a circle shape. It is used to define the boundary
    conditions of the PDE equations that are to be solved by the neural networks.
    Also it is used to generate sample points on the edge and in the domain of the
    circle and visualize the circle

    Parameters:
        seed (int): seed for random number generator
        center (np.ndarray): center of the circle
        radius (Number): radius of the circle
    """
    
    def __init__(
            self, 
            seed: int,
            edge_n: int,
            domain_n: int,
            cross_sample_generate: Number,
            center: np.ndarray, 
            radius: np.ndarray,
            even_sample: bool = True
        ) -> None:

        if not isinstance(radius, Number):
            raise TypeError('radius must be a number.')
        if radius <= 0:
            raise ValueError('radius must be positive.')

        super().__init__(
            seed,
            edge_n,
            domain_n,
            cross_sample_generate,
            center,
            radius,
            radius,
            even_sample
        )


__all__ = [
    'ElipseShape',
    'CircleShape'
]
