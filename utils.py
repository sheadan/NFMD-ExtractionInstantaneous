import numpy as np
from scipy.signal import hilbert


def get_noise(signal, SNR):
    #std_z = np.std(signal)
    noise = np.random.randn(*signal.shape)

    z_norm = np.linalg.norm(signal)
    SNR_correct_norm = np.sqrt((z_norm**2)/(10**(SNR/10)))

    noise_norm = np.linalg.norm(noise)
    noise = (SNR_correct_norm/noise_norm)*noise

    return noise


def compute_hilbert(signal_x, fs):
    analytic_signal = hilbert(signal_x, axis=0) #  N=signal_x.shape[-1]*10,
    amplitude_envelope = np.abs(analytic_signal)
    inst_phase = np.unwrap(np.angle(analytic_signal))
    inst_freq = (np.gradient(inst_phase, 1/fs)/(2.0*np.pi))
  
    return analytic_signal, inst_freq, inst_phase, amplitude_envelope

