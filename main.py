"""main.py

Script to generate windows, run simulation, and generate figures
according to specifications.

luizfelipe.coelho@smt.ufrj.br
Feb 17, 2024
"""


import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import cpu_count, Pool
from scipy.constants import speed_of_light
from src.window_manipulation import gen_win, reshape_win
from src.w_ofdm import simulation_fun


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, help='Operation mode.',
                        choices=['gen_win', 'run_sim', 'gen_figs'])
    parser.add_argument('-M', '--dft_len', type=int, help='Length of the DFT.',
                        default=256)
    parser.add_argument('-red', '--redundancy_len', type=float, default=10,
                        help='Length of redundancy as a percentage of the DFT.')
    parser.add_argument('-cs', '--cs_len', type=float, default=1/4,
                        help='Length of CS as a percentage of the CP length.',
                        choices=[1/2, 1/4, 1/8, 1/16, 0])
    parser.add_argument('--beta', type=float, default=.5,
                        help='Length of overlap and add as a ratio of redundancy',
                        choices=[1/2, 1/4, 1/8, 1/16])
    parser.add_argument('-win_type', '--window_type', type=str,
                        help='Window type for initial coefficients.',
                        choices=['bartlett', 'blackman', 'hamming', 'hanning',
                                 'kaiser', 'hann', 'rect'], default='blackman')
    parser.add_argument('--snr', type=str, default='-21,51,3',
                        help='SNR for the simulation. ([start, ]stop, [step).')
    parser.add_argument('--parallel', action=argparse.BooleanOptionalAction,
                        help='Wanna do some parallel computing?')
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    # Get arguments:
    args = arg_parser()
    # Define/calculate variables:
    red_len = round(args.dft_len * args.redundancy_len / 100)
    red_len = red_len + 1 if red_len % 2 else red_len
    cs_len = int(red_len * args.cs_len)
    cp_len = red_len - cs_len
    win_len = args.dft_len + cp_len + cs_len
    beta = cs_len if int(args.beta*(cp_len+cs_len)) > cs_len else \
        int(args.beta*(cp_len+cs_len))
    fft_len_plot = 2**12
    window_folder = 'windows/'
    os.makedirs(window_folder, exist_ok=True)
    print(f'Run w-OFDM using:\nREDUNDANCY:{red_len} -- CP:{cp_len}, CS:{cs_len}.')
    print(f'Overlap and add: {beta} samples.')
    carrier_frequency = 2*1e9
    sampling_period = 200*1e-9
    velocity = 120/3.6
    no_samples = 21
    no_symbols = 16
    dft_length = 256
    symbol_duration = dft_length*sampling_period
    frame_duration = no_symbols*symbol_duration
    doppler_frequency = (velocity/speed_of_light)*carrier_frequency
    snr_arr = np.arange(*[int(val) for val in args.snr.split(',')])

    match args.mode:
        case 'gen_win':
            # beta = 100
            Gi_sf = 40
            win_arr, win_init = gen_win(args.window_type, int(2*beta)+1, Gi_sf,
                                        3, .3)
            win_original = np.hstack((win_init[:beta],
                                      np.ones((win_len-int(2*beta),)),
                                      win_init[-beta:]))
            win_reshape = np.hstack((win_arr[:beta],
                                     np.ones((win_len-int(2*beta),)),
                                     win_arr[-beta:]))
            mm = 50
            win_reshape2 = reshape_win(win_original, Gi_sf, 3, .3)
            win_arr_freq = np.fft.fft(win_arr,
                                      fft_len_plot)[:int(fft_len_plot/2)]
            win_init_freq = np.fft.fft(win_init,
                                       fft_len_plot)[:int(fft_len_plot/2)]
            win_reshape_freq = np.fft.fft(win_reshape,
                                          fft_len_plot)[:int(fft_len_plot/2)]
            win_original_freq = np.fft.fft(win_original,
                                           fft_len_plot)[:int(fft_len_plot/2)]
            win_reshape2_freq = np.fft.fft(win_reshape2,
                                           fft_len_plot)[:int(fft_len_plot/2)]
            # Take a look:
            fig0 = plt.figure()
            ax0 = fig0.add_subplot(321)
            ax0.plot(win_arr, label='Red. Sidelobe')
            ax0.plot(win_init, label=args.window_type)
            ax1 = fig0.add_subplot(322)
            ax1.plot(20*np.log10(np.abs(win_arr_freq)))
            ax1.plot(20*np.log10(np.abs(win_init_freq)))
            ax2 = fig0.add_subplot(323)
            ax2.plot(win_reshape)
            ax2.plot(win_original)
            ax3 = fig0.add_subplot(324)
            ax3.plot(20*np.log10(np.abs(win_reshape_freq)))
            ax3.plot(20*np.log10(np.abs(win_original_freq)))
            ax4 = fig0.add_subplot(325)
            ax4.plot(win_reshape2)
            ax4.plot(win_original)
            ax5 = fig0.add_subplot(326)
            ax5.plot(20*np.log10(np.abs(win_reshape2_freq)))
            ax5.plot(20*np.log10(np.abs(win_original_freq)))
            ax0.legend()
            fig0.tight_layout()
            plt.show()
            # Save window coefficients:
            win_path = os.path.join(window_folder,
                                    f'win_{Gi_sf}_{win_len}.npy')
            
        case 'run_sim':

            data_list = [() for snr in snr_arr]
            if args.parallel:
                with Pool(cpu_count()) as pool:
                    pool.map(simulation_fun, data_list)
            else:
                [simulation_fun(data) for data in data_list]

        case 'gen_figs':
            pass
            