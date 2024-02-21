"""channel_modeling.py

Script with channel modeling methods.

luizfelipe.coelho@smt.ufrj.br
Feb 20, 2024
"""


import numpy as np


CHANNEL_ITUR = {'vehicularA': {
    'relative_delay': [0, 310*1e-9, 710*1e-9, 1090*1e-9, 1730*1e-9, 2510*1e-9],
    'avg_power': [0, -1, -9, -10, -15, -20],
    'doppler_spec': 6*['jakes']
}, 'vehicularB': {
    'relative_delay': [0, 300*1e-9, 8900*1e-9, 12900*1e-9, 17100*1e-9, 20000*1e-9],
    'avg_power': [-2.5, 0, -12.8, -10, -25.2, -16],
    'doppler_spec': 6*['jakes']
}, 'outdoor-indoorA': {
    'relative_delay': [0, 110*1e-9, 190*1e-9, 410*1e-9],
    'avg_power': [0, -9.7, -19.2, -22.8],
    'doppler_spec': 4*['jakes']
}, 'outdoor-indoorB': {
    'relative_delay': [0, 200*1e-9, 800*1e-9, 1200*1e-9, 2300*1e-9, 3700*1e-9],
    'avg_power': [0, -.9, -4.9, -8, -7.8, -23.9],
    'doppler_spec': 6*['jakes']
}}


def gen_chan(standard:str, no_samples:int, doppler_freq:float,
             sampling_rate:float, frame_duration:float, no_frames:int):
    """Generates a channel model following ITU-R recommendations.
    
    Parameters
    ----------
    standard : str
        ITU-R standard channel.
    no_samples : int
        Number of samples in tapped-delay line.
    doppler_freq : float
        Doppler frequency.
    sampling_rate : float
        Sampling rate of the channel model.
    frame_duration : float
        Duration of frame in which a channel model is valid before
        change in seconds.
    no_frames : int
        Number of frames sent.
    
    Returns
    -------
    channel_model : np.ndarray
        Array containing channel model according to specifications.
    """

    def read_channel_data(channel_model:dict):
        """Method to read data from dictionary."""

        return (list(zip(channel_model['relative_delay'],
                         channel_model['avg_power'],
                         channel_model['doppler_spec'])),
                len([doppler for doppler in channel_model['doppler_spec']
                     if doppler == 'jakes']))
    
    def adjust_power(power_db:float, rayleigh_waveform:np.ndarray):
        """Method to adjust waveform."""

        waveform_power = np.mean(
            np.dot(np.conj(rayleigh_waveform), rayleigh_waveform)
        )
        new_power = 10**(power_db/10)
        return np.sqrt(new_power/waveform_power)*rayleigh_waveform

    channel_model, no_jakes = read_channel_data(CHANNEL_ITUR[standard])
    rayleigh_waveforms = rayleigh_fading_gmeds_1(21, doppler_freq, no_frames,
                                                 1/frame_duration, no_jakes)
    sinc_mat = np.zeros((no_samples, len(channel_model)), dtype=np.complex128)
    coef_mat = np.zeros((len(channel_model), no_frames), dtype=np.complex128)
    base_time_axis = np.linspace(-(no_samples-1)/2, (no_samples+1)/2, no_samples)
    jakes_idx = 0
    for idx, data in enumerate(channel_model):
        sinc_mat[:, idx] = np.sinc(data[0]*sampling_rate - base_time_axis)
        if data[2] == 'jakes':
            coef_mat[idx, :] = \
                adjust_power(data[1], rayleigh_waveforms[:, jakes_idx])
            jakes_idx += 1
        else:
            # Must implement later one with LOS and flat.
            pass

    return np.matmul(sinc_mat, coef_mat)



def rayleigh_fading_gmeds_1(no_oscillators:int, doppler_freq:float,
                            no_channels:int, sampling_freq:float,
                            no_waveforms:int):
    """Method to generate Rayleigh fading waveform using the GMEDS_1
    algorithm described in:
    
    - M. Patzold, C. -X. Wang and B. O. Hogstad, "Two new
    sum-of-sinusoids-based methods for the efficient generation of
    multiple uncorrelated rayleigh fading waveforms," in IEEE
    Transactions on Wireless Communications, vol. 8, no. 6,
    pp. 3122-3131, June 2009, doi: 10.1109/TWC.2009.080769.
    
    Parameters
    ----------
    no_oscillators : int
        Number of oscillators (= 20 is good enough).
    doppler_freq : float
        Maximum Doppler frequency.
    sampling_freq : float
        Sampling rate.
    no_channels : int
        Number of channels in simulation.
    no_waveforms : int
        Total number of uncorrelated waveforms.
    
    Returns
    -------
    rayleigh_fading_waveforms : np.ndarray
        Rayleigh fading waveform.
    """

    time_axis = np.arange(0, no_channels, 1)/sampling_freq
    rayleigh_fading_waveforms = np.zeros((len(time_axis), no_waveforms),
                                         dtype=np.complex128)
    for wave_idx in range(no_waveforms):
        real_wave = np.zeros((len(time_axis),))
        imag_wave = np.zeros((len(time_axis),))
        for oscil_idx in range(no_oscillators):
            angle_rotation = (np.pi/(4*no_oscillators)) \
                * (wave_idx/(no_waveforms+2))  # Must be negative for imaginary
            angle_arrival = (np.pi/(2*no_oscillators))*(oscil_idx+.5)
            oscil_freq_real = doppler_freq*np.cos(angle_arrival+angle_rotation)
            oscil_freq_imag = doppler_freq*np.cos(angle_arrival-angle_rotation)
            real_wave += np.cos(2*np.pi*oscil_freq_real*time_axis
                                + np.pi*np.random.randn(1))
            imag_wave += np.cos(2*np.pi*oscil_freq_imag*time_axis
                                + np.pi*np.random.randn(1))
        rayleigh_wave = np.sqrt(2/no_oscillators)*(real_wave + 1j*imag_wave)
        rayleigh_fading_waveforms[:, wave_idx] = rayleigh_wave

    return rayleigh_fading_waveforms
