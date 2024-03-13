import json

import numpy as np


def make_noisy_state(state, noise_level, seed=0):
    rn_state = np.random.RandomState(seed)
    flip_mask = rn_state.rand(*state.shape) < noise_level
    return np.logical_xor(state, flip_mask).astype(int)


def load_letters_from_json(filename):
    with open(filename, 'r') as file:
        letters = json.load(file)
    return letters


def generate_binary_matrix(letter, letters_file='letters.json'):
    # Load letters from JSON file
    letters = load_letters_from_json(letters_file)

    # Get the binary representation of the letter
    binary_representation = letters.get(letter.upper(), None)

    if binary_representation is None:
        raise ValueError(f'Letter {letter} not found in the letters file')

    # Convert binary representation to matrix
    binary_matrix = []
    for row in binary_representation:
        binary_row = [int(pixel) for pixel in row]
        binary_matrix.append(binary_row)

    return np.array(binary_matrix)


def string_to_matrix(string, letters_file='letters.json'):
    matrices = [generate_binary_matrix(letter, letters_file) for letter in string.upper().replace(' ', '_')]
    matrices = [np.concatenate((letter, np.zeros((letter.shape[0], 1))), axis=1) for letter in matrices]
    return np.concatenate(matrices, axis=1)


def strings_to_matrix(strings, letters_file='letters.json'):
    mat_list = [string_to_matrix(s, letters_file) for s in strings]
    max_columns = max(m.shape[-1] for m in mat_list)
    mat_list_padded = []
    for m in mat_list:
        mat = np.zeros((m.shape[0], max_columns))
        mat[:m.shape[0], :m.shape[1]] = m
        mat_list_padded.append(mat)
        mat_list_padded.append(np.zeros((1, max_columns)))
    return np.concatenate(mat_list_padded, axis=0)
