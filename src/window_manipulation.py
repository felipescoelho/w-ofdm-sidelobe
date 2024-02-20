"""window_manipulation.py

Script to manipulate window coefficients.

luizfelipe.coelho@smt.ufrj.br
Feb 17, 2024
"""


import numpy as np
import scipy as sp
from subprocess import Popen, PIPE


def out_parser(txt_out: str, N:int):
    """Method to parse output from FORTRAN."""
    out_list = txt_out.split('\n')
    idx = out_list.index(' COEFFICIENT VALUES')
    out_list_clean = out_list[idx+1:idx+1+N]
    out_vect = np.array([float(coeff[-12:]) for coeff in out_list_clean])
    
    return out_vect


def gen_win(win_type: str, N: int, dB_in: float, Gi_sf: float, 
            Gi_sf_sc: float):
    """
    Method to generate coefficients for window with reduced sidelobe
    maginitude.
    This method wraps the FORTRAN compiled code for window manipulation.
    
    Parameters
    ----------
    win_type : str
        Window type.
    N : int
        Number of coefficients.
    dB_in : float
        Desired sidelobe attenuation in dB.
    Gi_sf : float
        Initial gain increment scale factor (Gi_sf > 1).
    Gi_sf_sc : float
        Adjustment of the intial gain increment in case of overshoot.
        (0 < Gi_sf_sc < 1). -- Gi_sf = 1 + (Gi_sf - 1) * Gi_sf_sc.
    
    Returns
    -------
    w_arr : np.ndarray
        Array containing window coefficients.
    """

    aux_idx = int(N/2) if N % 2 == 0 else int((N+1)/2)
    input_list = [str(N), str(dB_in), str(Gi_sf), str(Gi_sf_sc)]

    match win_type:
        case 'bartlett':
            w_i = np.bartlett(N)
        case 'blackman':
            w_i = np.abs(np.blackman(N))
        case 'hamming':
            w_i = np.hamming(N)
        case 'hanning':
            w_i = np.hanning(N)
        case 'kaiser':
            w_i = np.kaiser(N, 14)
        case 'hann':
            w_i = sp.signal.windows.hann(N)
        case 'rect':
            w_i = np.ones((N,))

    with Popen('./a.out', stdin=PIPE, stdout=PIPE) as proc:
        input_str = ','.join(input_list)+'\n'
        proc.stdin.write(input_str.encode())
        proc.stdin.flush()
        win2str = [f'{coef:.16f}' for coef in w_i[:aux_idx]]
        input_win = '\n'.join(win2str)+'\n'
        proc.stdin.write(input_win.encode())
        proc.stdin.flush()
        out, error = proc.communicate()
    out_str = out.decode('utf-8')
    w_arr = out_parser(out_str, N)
    w_arr /= max(w_arr)

    return w_arr, w_i


def reshape_win(win_arr: np.ndarray, dB_in: float, Gi_sf: float,
                Gi_sf_sc: float):
    """
    Method to reshape window using window scaling for sidelobe magnitude
    reduction.

    This method wraps the FORTRAN compiled code for window manipulation.

    Parameters
    ----------
    win_arr : np.ndarray
        Window coefficients.
    dB_in : float
        Desired sidelobe attenuation in dB.
    Gi_sf : float
        Initial gain increment scale factor (Gi_sf > 1).
    Gi_sf_sc : float
        Adjustment of the intial gain increment in case of overshoot.
        (0 < Gi_sf_sc < 1). -- Gi_sf = 1 + (Gi_sf - 1) * Gi_sf_sc.
    
    Returns
    -------
    win_arr_out : np.ndarray
        Array containing window coefficients.
    """

    N = len(win_arr)
    aux_idx = int(N/2) if N % 2 == 0 else int((N+1)/2)
    input_list = [str(N), str(dB_in), str(Gi_sf), str(Gi_sf_sc)]
    with Popen('./a.out', stdin=PIPE, stdout=PIPE) as proc:
        input_str = ','.join(input_list)+'\n'
        proc.stdin.write(input_str.encode())
        proc.stdin.flush()
        win2str = [f'{coef:.16f}' for coef in win_arr[:aux_idx]]
        input_win = '\n'.join(win2str)+'\n'
        proc.stdin.write(input_win.encode())
        proc.stdin.flush()
        out, _ = proc.communicate()
    out_str = out.decode('utf-8')
    win_arr_out = out_parser(out_str, N)
    win_arr_out /= max(win_arr_out)

    return win_arr_out