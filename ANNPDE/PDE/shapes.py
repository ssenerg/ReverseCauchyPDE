from numpy import random as numpy_random
from .utilities import int_validator
from abc import ABC, abstractmethod
import plotly.graph_objects as go
from scipy.stats.qmc import Sobol
from typing import Tuple, List
import random as python_random
from numbers import Number
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
            even_sample: bool,
            _edge_zero: bool = False
        ) -> None:
        
        if not int_validator(seed):
            raise TypeError('seed must be an integer.')
        elif not 0 <= seed <= 2 ** 32 - 1:
            raise ValueError('seed must be in range [0, 2**32 - 1].')

        if not int_validator(edge_n):
            raise TypeError('edge_n must be an integer.')
        elif edge_n <= 0:
            if not _edge_zero or edge_n != 0:
                raise ValueError('edge_n must be positive.') 
            
        elif not int_validator(domain_n):
            raise TypeError('domain_n must be an integer.')
        elif domain_n <= 0:
            raise ValueError('domain_n must be positive.')
        elif not isinstance(cross_sample_generate, Number):
            raise TypeError('cross_sample_generate must be a number.')
        elif cross_sample_generate < 1:
            raise ValueError('cross_sample_generate must be greater than 1.')
        elif not isinstance(even_sample, bool):
            raise TypeError('even_sample must be a boolean.')
        
        numpy_random.seed(seed)
        python_random.seed(seed)

        self.edge_n = edge_n
        self.domain_n = domain_n
        self.cross_sample = cross_sample_generate

        self.edge_sample = None
        self.domain_sample = None
        self.even_sample = even_sample

    def _edge_sample(self) -> np.ndarray:

        """
        This function is used to get the sample points on the edge of the shape

        --------------------------------------------------------------------------------

        Returns:
            edge_sample (np.ndarray): sample points on the edge of the shape
        """
        
        self.edge_sample = np.array([])
        if self.even_sample:
            self._even_edge_sample()
        else:
            self._instant_edge_sample()
    
    def _domain_sample(self) -> np.ndarray:

        """
        This function is used to get the sample points in the domain of the shape

        --------------------------------------------------------------------------------

        Returns:
            domain_sample (np.ndarray): sample points in the domain of the shape
        """
        
        self.domain_sample = np.array([])
        if self.even_sample:
            self._even_domain_sample()
        else:
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
    
    @abstractmethod
    def _even_domain_sample(self) -> None:
        pass

    @abstractmethod
    def _even_edge_sample(self) -> None:
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
    


    def plot(self, size: int = 1) -> None:
        
        if self.edge_sample is None:
            self._edge_sample()
            self._domain_sample()

        if self.dim == 1:
            self._plot_1d(size)
            return
        
        trace1 = go.Scatter(
            x=self.edge_sample[:, 0],  # X-values for edge_sample
            y=self.edge_sample[:, 1],  # Y-values for edge_sample
            mode='markers',
            marker=dict(color='blue', size=size),  # Change color as needed
            name='Edge Sample'
        )

        trace2 = go.Scatter(
            x=self.domain_sample[:, 0],  # X-values for domain_sample
            y=self.domain_sample[:, 1],  # Y-values for domain_sample
            mode='markers',
            marker=dict(color='red', size=size),  # Change color as needed
            name='Domain Sample'
        )

        # Combine the plots
        data = [trace1, trace2]

        # Define layout
        layout = go.Layout(
            title='Scatter Plot of Samples',
            xaxis=dict(
                title='X-axis',
                # Constrain the aspect ratio to ensure equal scaling
                constrain='domain'
            ),
            yaxis=dict(
                title='Y-axis',
                # Set scale anchor to 'x' to force y-axis to scale with x-axis
                scaleanchor='x',
                scaleratio=1
            ),
            showlegend=True
        )

        # Create the figure
        fig = go.Figure(data=data, layout=layout)

        # Show the figure
        fig.show()
            
    def _plot_1d(self, size: int = 1) -> None:

        trace1 = go.Scatter(
            x=self.domain_sample,
            y=np.zeros(self.domain_n),
            mode='markers',
            marker=dict(color='red', size=size),
            name='Samples'
        )

        data = [trace1]

        layout = go.Layout(
            title='Scatter Plot of Samples',
            xaxis=dict(
                title='t-axis',
            ),
            yaxis=dict(
                title='',
            ),
            showlegend=True
        )

        fig = go.Figure(data=data, layout=layout)

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
        edge_n (int): number of sample points on the edge of the elipse
        domain_n (int): number of sample points in the domain of the elipse
        cross_sample_generate (Number): number of sample points generated for each
        center (np.ndarray): center of the elipse
        v_radius (Number): vertical radius of the elipse
        h_radius (Number): horizontal radius of the elipse
        edge_cuts_angle (List[Tuple[Number, Number]]): list of tuples of angles in 
        radians
        even_sample (bool): if True, the sample points are generated evenly
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
            edge_cuts_angle: List[Tuple[Number, Number]] = None,
            even_sample: bool = True
        ) -> None:

        super().__init__(
            seed,
            edge_n,
            domain_n,
            cross_sample_generate,
            even_sample
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
        if edge_cuts_angle is not None:
            # cluster of [0, 2 * pi] like [(0, pi / 2),  (pi, 3 * pi / 2)]
            if not isinstance(edge_cuts_angle, list):
                raise TypeError('edge_cuts_angle must be a list.')
            if not all(isinstance(item, tuple) for item in edge_cuts_angle):
                raise TypeError('edge_cuts_angle must be a list of tuples.')
            if not all(isinstance(item[0], Number) for item in edge_cuts_angle):
                raise TypeError('edge_cuts_angle must be a list of tuples of numbers.')
            if not all(isinstance(item[1], Number) for item in edge_cuts_angle):
                raise TypeError('edge_cuts_angle must be a list of tuples of numbers.')
            if not all(0 <= item[0] <= 2 * np.pi for item in edge_cuts_angle):
                raise ValueError('edge_cuts_angle must be a list of tuples of numbers in range [0, 2 * pi].')
            if not all(0 <= item[1] <= 2 * np.pi for item in edge_cuts_angle):
                raise ValueError('edge_cuts_angle must be a list of tuples of numbers in range [0, 2 * pi].')
            if not all(item[0] < item[1] for item in edge_cuts_angle):
                raise ValueError('edge_cuts_angle must be a list of tuples of numbers in range [0, 2 * pi].')
            if not all(item[0] < item[1] for item in edge_cuts_angle):
                raise ValueError('edge_cuts_angle must be a list of tuples of numbers in range [0, 2 * pi].')
            
            # now merge the intervals which are overlapping
            edge_cuts_angle.sort(key=lambda x: x[0])
            merged = []
            for item in edge_cuts_angle:
                if not merged or merged[-1][1] < item[0]:
                    merged.append(item)
                else:
                    merged[-1][1] = max(merged[-1][1], item[1])
        else:
            edge_cuts_angle = [(0, 2 * np.pi)]

        total_cut = sum(map(lambda x: x[1] - x[0], edge_cuts_angle))
        edge_cut_samples_count = list(
            map(
                lambda x: int(((x[1] - x[0]) / total_cut) * self.edge_n), 
                edge_cuts_angle
            )
        )
        edge_cut_samples_count[-1] = self.edge_n - sum(edge_cut_samples_count[:-1])

        self.eca = zip(edge_cuts_angle, edge_cut_samples_count)

        self.center = center
        self.v_radius = v_radius
        self.h_radius = h_radius
        if even_sample:
            self.sobol_engine = Sobol(d=2, scramble=True)
        self.dim = self._dim()

    def _dim(self) -> int:

        """
        This function is used to get the dimension of the Elipse / Circle
        """

        length = len(self.center)
        if length != 2:
            raise ValueError('center must be a 2D vector.')
        return length
    
    def _sobol_sampling(self, n_samples: int) -> np.ndarray:
        """
        Generate Sobol sequence samples within the unit square and scale them accordingly.
        
        Parameters:
            n_samples (int): The number of samples to generate.
            
        Returns:
            np.ndarray: Sobol sequence samples scaled to the appropriate size.
        """
        samples = self.sobol_engine.random(n=n_samples)
        return samples

    def _even_domain_sample(self) -> None:

        # Generate Sobol sequence samples
        sobol_samples = self._sobol_sampling(self.domain_n)
        # Scale samples to fit the ellipse
        theta = sobol_samples[:, 0] * 2 * np.pi
        radii = np.sqrt(sobol_samples[:, 1])  # Use square root to ensure uniform distribution
        h_radii = radii * self.h_radius
        v_radii = radii * self.v_radius

        x = self.center[0] + h_radii * np.cos(theta)
        y = self.center[1] + v_radii * np.sin(theta)

        self.domain_sample = np.column_stack((x, y))

    def _even_edge_sample(self) -> None:

        for interv, count in self.eca:
            # Generate Sobol sequence samples
            sobol_samples = self._sobol_sampling(count)
            # Scale samples to fit the ellipse
            theta = sobol_samples[:, 0] * (interv[1] - interv[0]) + interv[0]
            h_radius = self.h_radius
            v_radius = self.v_radius

            x = self.center[0] + h_radius * np.cos(theta)
            y = self.center[1] + v_radius * np.sin(theta)

            if self.edge_sample is None or self.edge_sample.shape[0] == 0:
                self.edge_sample = np.column_stack((x, y))
            else:
                self.edge_sample = np.concatenate(
                    (self.edge_sample, np.column_stack((x, y)))
                )

    def _instant_domain_sample(self) -> None:

        theta = np.random.uniform(0, 2 * np.pi, self.domain_n)
        h_radius = np.random.uniform(0, self.h_radius, self.domain_n)
        v_radius = np.random.uniform(0, self.v_radius, self.domain_n)

        x = self.center[0] + h_radius * np.cos(theta)
        y = self.center[1] + v_radius * np.sin(theta)

        self.domain_sample = np.column_stack((x, y))

    def _instant_edge_sample(self) -> None:

        for interv, count in self.eca:
            theta = np.random.uniform(interv[0], interv[1], count)
            h_radius = self.h_radius
            v_radius = self.v_radius

            x = self.center[0] + h_radius * np.cos(theta)
            y = self.center[1] + v_radius * np.sin(theta)

            if self.edge_sample is None:
                self.edge_sample = np.column_stack((x, y))
            else:
                self.edge_sample = np.concatenate(
                    (self.edge_sample, np.column_stack((x, y)))
                )


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
            edge_cuts_angle: List[Tuple[Number, Number]] = None,
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
            edge_cuts_angle,
            even_sample
        )


class LineShape(BaseShape):

    def __init__(
            self,
            seed: int,
            n: int,
            cross_sample_generate: Number,
            start_point: Number,
            end_point: Number,
            even_sample: bool = True
        ) -> None:

        super().__init__(
            seed,
            0,
            n,
            cross_sample_generate,
            even_sample,
            _edge_zero=True
        )

        if not isinstance(start_point, Number):
            raise TypeError('start_point must be a number.')
        if not isinstance(end_point, Number):
            raise TypeError('end_point must be a number.')
        
        self.start_point = start_point
        self.end_point = end_point
        self.dim = self._dim()
        if even_sample:
            self.sobol_engine = Sobol(d=1, scramble=True)
    
    def _even_domain_sample(self) -> None:

        scaled_samples = self.start_point + (self.end_point - self.start_point) * \
            self.sobol_engine.random(n=self.domain_n)
        self.domain_sample = np.sort(scaled_samples.flatten())

    def _even_edge_sample(self) -> None:
        return None

    def _instant_domain_sample(self) -> None:
            
        self.domain_sample = np.sort(
            np.random.uniform(self.start_point, self.end_point, self.domain_n)
        )

    def _instant_edge_sample(self) -> None:
                
        return None

    def _dim(self) -> int:
            
        """
        This function is used to get the dimension of the Line
        """

        return 1

    def get(self) -> Tuple[np.ndarray, np.ndarray]:

        """
        This function is used to get the sample points of the shape

        --------------------------------------------------------------------------------

        Returns:
            edge_sample (np.ndarray): sample points on the edge of the shape
            domain_sample (np.ndarray): sample points in the domain of the shape
        """

        self._domain_sample()

        return self.domain_sample
    

__all__ = [
    'ElipseShape',
    'CircleShape',
    'LineShape'
]
