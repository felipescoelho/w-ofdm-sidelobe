"""utils.py

Script with utilitary methods for simulation and stuff.

luizfelipe.coelho@smt.ufrj.br
Jan 20, 2024
"""


import numpy as np


def gen_symbols(mod_size: int, N: int, no_symbols: int) -> np.ndarray:
    """Method to generate QAM symbols."""

    match mod_size:
        case 4:
            symbols = (-1-1j, -1+1j, 1-1j, 1+1j)
        case 16:
            symbols = (-3-3j, -3-1j, -3+1j, -3+3j, -1-3j, -1-1j, -1+1j,
                        -1+3j, 1-3j, 1-1j, 1+1j, 1+3j, 3-3j, 3-1j, 3+1j,
                        3+3j)
    dig_symbols = np.random.choices(symbols, size=(N, no_symbols),
                                    replace=True)

    return dig_symbols


def awgn(x: np.ndarray, snr: float) -> np.ndarray:
    """Method to add white Gaussian noise adjusted by SNR.
    
    Parameters
    ----------
    x : np.ndarray
        Signal to be poluted with noise.
    snr : float
        Signal to noise ratio in dB.
    
    Returns
    -------
    y : np.ndarray
        Noisy signal.
    """

    Px = np.vdot(x, x)
    n = np.random.randn(len(x),) + 1j*np.random.randn(len(x),)
    Pn = np.vdot(n, n)
    n_adjusted = np.sqrt(Px*10**(-.1*snr)/Pn)*n

    return x + n_adjusted


def decision(x:np.ndarray, symbols:np.ndarray) -> np.ndarray:
    """Method to decide which symbol is received.
    
    Criterion: Shortest distance.
    
    Parameters
    ----------
    x : np.ndarray
        Received signal.
    symbols : np.ndarray
        Possible values for reception.
    
    Returns
    -------
    y : np.ndarray

    """

    n_rows, n_cols = x.shape
    y = np.zeros((n_rows, n_cols), dtype=np.complex128)
    for i in range(n_rows):
        for j in range(n_cols):
            min_idx = np.argmin(np.abs(symbols - x[i, j]))
            y[i, j] = symbols[min_idx]

    return y


def vect_ov(frame: np.ndarray, beta: int):
    """Performs the vectorization with overlap and add."""

    frame_ov = frame[:, beta:]
    frame_ov[:-1, -beta:] += frame[1:, :beta]

    return np.hstack((frame[0, :beta], frame_ov.flatten()))
