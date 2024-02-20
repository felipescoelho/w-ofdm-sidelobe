"""w_ofdm.py

Script with code for w-OFDM using a pre-defined window.

luizfelipe.coelho@smt.ufrj.br
Feb 19, 2024
"""


import numpy as np
from numba import njit
from utils import gen_symbols, awgn, decision, vect_ov


def simulation_fun(**kwargs):
    """Method to run simulation using our wOFDM class and parallel
    computing if needed.
    
    Parameters
    ----------
    Keyword Arguments:
    
    Returns
    -------
    """

    
    ofdm_model = wOFDM(**kwargs)
    ofdm_model.run_simulation()


class wOFDM:
    """Class to implement w-OFDM system."""

    @staticmethod
    @njit(fastmath=True)
    def __run_simulation(tx_mat: np.ndarray, rx_mat: np.ndarray,
                         no_symbols: int, ensemble: int, snr_arr: np.ndarray,
                         channels: np.ndarray):
        """"""
        ser = np.zeros((len(snr_arr),), dtype=np.float64)
        for idx, snr in enumerate(snr_arr):
            results = 0
            for ch_idx in range(channels.shape[1]):
                results_mc = 0
                for _ in range(ensemble):
                    symbols = gen_symbols(16, tx_mat.shape[1], no_symbols)


    @staticmethod
    def __idft_mat(N: int):
        """Matrix for unitary IDFT."""

        idft_mat = np.zeros((N, N), dtype=np.complex128)
        for n in range(N):
            for k in range(N):
                idft_mat[n, k] = np.exp(1j*2*np.pi*n*k/N)/np.sqrt(N)
        
        return idft_mat
    
    @staticmethod
    def __gamma_mat(N: int, mu: int, rho: int):
        """Matrix to add redundancy."""

        gamma_mat = np.vstack((
            np.hstack((np.zeros((mu, N-mu)), np.eye(mu))),
            np.eye(N),
            np.hstack((np.eye(rho), np.zeros((rho, N-rho))))
        ))

        return gamma_mat
    
    @staticmethod
    def __circ_shift_mat(N: int, beta: int):
        """Matrix to perform circular shift."""

        circ_shift_mat = np.vstack((
            np.hstack((np.zeros((N-beta, beta)), np.eye(N-beta))),
            np.hstack((np.eye(beta), np.zeros((beta, N-beta))))
        ))

        return circ_shift_mat
    
    @staticmethod
    def __dft_mat(N: int):
        """Matrix for unitary DFT."""

        dft_mat = np.zeros((N, N), dtype=np.complex128)
        for n in range(N):
            for k in range(N):
                dft_mat[n, k] = np.exp(-1j*2*np.pi*n*k/N)/np.sqrt(N)
        
        return dft_mat

    def __init__(self, **kwargs):
        """"""

        self.N = kwargs['dft_len']
        self.mu = kwargs['cp_len']
        self.rho = kwargs['cs_len']
        self.beta = kwargs['overlap_len']
        self.mod_size = kwargs['mod_size']

    def run_simulation(self, mode: str):
        """Method to run w-OFDM system simulation."""
        
