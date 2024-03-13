import pytest
import numpy as np
import sys
import os
try:
    from src.hopfield import HopfieldNetwork, visualize_state
    from src.utils import generate_binary_matrix, make_noisy_state
except ModuleNotFoundError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
    from src.hopfield import HopfieldNetwork
    from src.utils import generate_binary_matrix, make_noisy_state

TIME_LIMIT = 300
message = np.load(os.path.join(os.path.dirname(__file__), "data", "message.npy"), allow_pickle=True)
image = np.load(os.path.join(os.path.dirname(__file__), "data", "image.npy"), allow_pickle=True)
letters_file = os.path.join(os.path.dirname(__file__), "data", "big_letters.json")


@pytest.mark.timeout(TIME_LIMIT)
@pytest.mark.parametrize(
    "state, noise_level, seed, iteration, expected",
    [
        (generate_binary_matrix(letter, letters_file), noise, seed, 1_000, 0.9)
        for letter in [
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
        ]
        for noise in [0.1, 0.2]
        for seed in [0, 42]
    ],
)
def test_easy_hopfield(state, noise_level, seed, iteration, expected):
    noisy_state = make_noisy_state(state, noise_level, seed)
    net = HopfieldNetwork(
        weights=np.zeros((state.size, state.size)),
        thresholds=np.zeros(state.size),
        state=noisy_state.astype(np.float64),
    )
    net.learn_one_state(state)
    net.simulate(iteration, noisy_state, 2)
    accuracy = np.mean(np.isclose(net.state, state))
    reversed_accuracy = np.mean(np.isclose(net.state, 1 - state))
    assert accuracy >= expected or reversed_accuracy >= expected


@pytest.mark.timeout(TIME_LIMIT)
@pytest.mark.parametrize(
    "state, noise_level, seed, iteration, expected",
    [
        (state, noise, seed, itr, 0.9)
        for acc, state, itr in zip([0.9, 0.9], [message, image], [30_000, 20_000])
        for noise in [0.1, 0.2, 0.25]
        for seed in [0, 1, 2]
    ],
)
def test_hard_hopfield(state, noise_level, seed, iteration, expected):
    noisy_state = make_noisy_state(state, noise_level, seed)
    net = HopfieldNetwork(
        weights=np.zeros((state.size, state.size)),
        thresholds=np.zeros(state.size),
        state=noisy_state.astype(np.float64),
    )
    net.learn_one_state(state)
    net.simulate(iteration, noisy_state, 2)
    accuracy = np.mean(np.isclose(net.state, state))
    reversed_accuracy = np.mean(np.isclose(net.state, 1 - state))
    assert accuracy >= expected or reversed_accuracy >= expected


@pytest.mark.timeout(TIME_LIMIT)
@pytest.mark.parametrize(
    "state, iteration",
    [
        (generate_binary_matrix(letter, letters_file), 1_000)
        for letter in [
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
        ]
    ],
)
def test_hopfield_energy(state, iteration):
    initial_state = np.zeros_like(state)
    net = HopfieldNetwork(
        weights=np.zeros((state.size, state.size)),
        thresholds=np.zeros(state.size),
        state=initial_state.astype(np.float64),
    )
    net.learn_one_state(state)
    history = net.simulate(iteration, initial_state, iteration)
    energies = [net.get_state_energy(state) for state in history]
    assert energies[-1] <= energies[0], (
        f"The energy of the final state {energies[-1]} is greater than the initial state {energies[0]}"
    )
    diff = np.diff(energies)
    neg_ratio = np.mean(diff <= 0)
    assert neg_ratio >= 0.99, f"The ration of null or negative energy differences is {neg_ratio} and should be ~1.0"
