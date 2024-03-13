import os
from typing import Optional, List
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


class HopfieldNetwork:
    r"""
    This class represents a Hopfield network, a type of recurrent neural network used for associative memory.

    :param weights: The weight matrix of the network, where W.shape[0] equals the number of neurons.
    :type weights: np.ndarray
    :param thresholds: A 1D numpy array containing the threshold value for each neuron.
    :type thresholds: np.ndarray
    :param state: The initial state of the network as a 2D numpy array of 0s and 1s.
    :type state: np.ndarray

    :ivar weights: The weight matrix of the network, where W.shape[0] equals the number of neurons.
    :vartype weights: np.ndarray
    :ivar thresholds: A 1D numpy array containing the threshold value for each neuron.
    :vartype thresholds: np.ndarray
    :ivar state: The state of the network as a 2D numpy array of neuron states.
    :vartype state: np.ndarray

    :raises ValueError: If the weight matrix is not square or if the weight matrix and the threshold vector do not
        have the same size.
    :raises ValueError: If the size of the state and the number of neurons do not match.
    """
    def __init__(
            self,
            weights: np.ndarray,
            thresholds: np.ndarray,
            state: Optional[np.ndarray] = None,
    ):
        raise NotImplementedError("This class is not implemented yet")

    @property
    def n(self) -> int:
        r"""
        The number of neurons in the network.

        :return: The number of neurons in the network.
        :rtype: int
        """
        raise NotImplementedError("This class is not implemented yet")

    @property
    def energy(self) -> float:
        """
        Calculates the energy of a Hopfield network for a given state.

        This function computes the energy of a state `x` in a Hopfield network, using the network's weight matrix `W`
        and the threshold values `theta`. The energy function is a measure of the stability of the state within the network,
        where lower energy states are typically more stable and represent memorized patterns.

        :return: The energy of the state `x` within the network. Lower values indicate more stable states.
        :rtype: float

        Note:
        - The energy function used here is based on the Hopfield network's definition, incorporating both
          the interactions between neurons (through the weight matrix `W`) and the effect of individual neuron thresholds.
        - This function returns a single scalar value representing the energy, making it useful for analyzing
          the stability of different states or for tracking the network's evolution over time.
        """
        raise NotImplementedError("This class is not implemented yet")

    def get_state_energy(self, state: np.ndarray) -> float:
        r"""
        Calculates the energy of a Hopfield network for a given state.

        :param state: The state of the network as a 2D numpy array of neuron states.
        :type state: np.ndarray

        :return: The energy of the state `x` within the network. Lower values indicate more stable states.
        :rtype: float
        """
        raise NotImplementedError("This class is not implemented yet")

    def set_state(self, state: np.ndarray):
        r"""
        Sets the state of the network to a given state.

        :param state: The state of the network as a 2D numpy array of neuron states.
        :type state: np.ndarray
        :return: None
        """
        raise NotImplementedError("This class is not implemented yet")

    def update(self) -> np.ndarray:
        r"""
        Updates the state of the network using a stochastic asynchronous update rule.

        :return: The updated state of the network as a 2D numpy array of neuron states.
        """
        raise NotImplementedError("This class is not implemented yet")

    def simulate(
            self,
            m: int,
            state: Optional[np.ndarray] = None,
            history_length: Optional[int] = 100,
    ) -> np.ndarray:
        """
        Simulates the evolution of states in a Hopfield network over time.

        This function iteratively updates the state of the network over `m` time steps, following a stochastic asynchronous update rule.
        At each time step, a single neuron is randomly selected and its state is updated based on the current global state of the network,
        the neuron's threshold, and the weight matrix `W`.

        :param m: The number of time steps over which the network's state will be evolved.
        :type m: int
        :param state: The initial state of the network as a 2D numpy array of neuron states.
        :type state: np.ndarray
        :param history_length: The number of states to record in the history. If None, all states are recorded.
        :type history_length: int

        :return: A 3D numpy array of shape (history_length, state.shape[0], state.shape[1]) containing the network's
            state at each recorded time step.
        :rtype: np.ndarray

        Note:
        - The network uses a stochastic asynchronous update rule, meaning at each time step, only one neuron's
          state is updated based on its input from other neurons and its threshold.
        - This simulation does not necessarily converge to a stable state within `m` steps,
          as the network's dynamics depend on the initial conditions, the structure of the weight matrix `W`,
          and the thresholds `theta`.
        """
        raise NotImplementedError("This class is not implemented yet")

    def learn_one_state(self, target_state: np.ndarray):
        """
        Adjusts the weights of a Hopfield network to memorize a single state.

        This function applies the Hebbian learning rule to adjust the weights of the network based on the provided state `y`.
        It strengthens the connections between neurons that are both active or both inactive in the state `y`.

        :param target_state: The state that the network should memorize, as a 2D numpy array of 0s and 1s.
        :type target_state: np.ndarray

        :return: The updated weight matrix after applying the learning rule for the given state `y`.
        :rtype: np.ndarray

        Note:
        - The diagonal elements of the weight matrix `W` are not modified, preserving the constraint that neurons do not have self-connections.
        - The state `y` is expected to be a binary vector where each element is either 0 or 1.
        """
        raise NotImplementedError("This class is not implemented yet")

    def learn_many_states(self, target_states: List[np.ndarray]):
        """
        Adjusts the weights of a Hopfield network to memorize multiple states.

        This function iterates over each state in `Y` and applies the `learn_one_state` function to adjust the network's weight matrix `W` for memorizing each state.
        It uses Hebbian learning to strengthen the connections between neurons based on their activity across all the provided states.

        :param target_states: A list of states that the network should memorize, where each state is a 2D numpy array of 0s and 1s.
        :type target_states: List[np.ndarray]

        :return: The updated weight matrix after applying the learning rule for all the given states in `Y`.
        :rtype: np.ndarray

        Note:
        - It is assumed that all states in `Y` are binary vectors of the same length as the number of neurons in the network.
        - The function updates the weights in-place and returns the same weight matrix object with updated values.
        """
        raise NotImplementedError("This class is not implemented yet")


def visualize_weights(net: HopfieldNetwork):
    fig, ax = plt.subplots()
    custom_cmap = ListedColormap(['deepskyblue', 'orangered'])
    ax.imshow(net.weights, cmap=custom_cmap)
    plt.show()
    return fig, ax


def visualize_state(net: HopfieldNetwork):
    fig, ax = plt.subplots()
    ax.axis('off')
    custom_cmap = ListedColormap(['deepskyblue', 'orangered'])
    ax.imshow(net.state, cmap=custom_cmap)
    plt.show()
    return fig, ax


def make_animation_from_history(history: np.ndarray, filename: str = "hopfield.gif", **kwargs):
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()
    ax.axis('off')
    custom_cmap = ListedColormap(['deepskyblue', 'orangered'])
    im = ax.imshow(history[0], cmap=custom_cmap)

    def update(frame):
        im.set_data(history[frame])
        return [im]

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    ani = FuncAnimation(fig, update, frames=range(len(history)), blit=True, interval=kwargs.get("interval", 200))
    ani.save(filename, writer='imagemagick', fps=kwargs.get("fps", None))
    return fig, ax, ani

