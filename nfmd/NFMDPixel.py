import numpy as np
import torch
from .NFMD import NFMD


class NFMDMode:
    '''
    Initialize an NFMDMode object containing the data relevant to
    a Fourier mode in the data. This is a data storage object.
    Parameters
    ----------
    IF: float
        frequency vector of instantaneous frequencies, one for each time in t
    IA: numpy.ndarray
        vector of instantaneous amplitudes
    A: numpy.ndarray
        vector of fourier mode coefficients
        (note: different from amplitude, but related)
    t: numpy.ndarray
        time vector, same length as the IF and IA vectors.
    '''
    def __init__(self, IF, IA, A, t):
        self.IF = IF
        self.IA = IA
        self.A = A
        self.t = t


class NFMDPixel:
    '''
    Initialize an NFMDPixel object for decomposing a signal into Fourier modes.
    Parameters
    ----------
    signal: numpy.ndarray
        Temporal signal to be analyzed.
    nfmd_options: dict
        options passed to the NFMD analysis class
    '''
    def __init__(self, signal,
                 nfmd_options={'num_freqs':3,
                               'window_size':320}):
        # Signal
        self.signal = signal

        #self.signal = (signal-np.mean(signal))
        #self.signal /= np.std(signal)

        # Signal Decomposition options
        self.nfmd_options = nfmd_options


    def analyze(self, dt=1):
        '''
        Initialize an NFMDMode object containing the data relevant to
        a Fourier mode in the data. This is a data storage object.
        Parameters
        ----------
        dt: float, optional
            timestep between datapoints in the signal array
        '''
        # Initialize the NFMD object
        nfmd = NFMD(self.signal, **self.nfmd_options)
        t = np.arange(nfmd.n)*dt

        # Decompose the signal using NFMD
        freqs, A, losses, indices = nfmd.decompose_signal()
    
        # Compute corrected frequencies (scaled by dt) and instantaneous amplitudes
        freqs = nfmd.correct_frequencies(dt=dt)
        amps = nfmd.compute_amps()

        # Slice for each window, and then the center point of each window!
        self.indices = indices
        self.mid_idcs = nfmd.mid_idcs

        # Compute frequencies, amplitudes and mean
        self.mean = nfmd.compute_mean()
        self.mean_t = t[nfmd.mid_idcs]

        # Organize the other modes into Mode objects:
        # Compute mean IF for each mode over entire time vector
        mean_freqs = np.mean(freqs, axis=0)

        # Organize the other modes:
        self.modes = []
        for i in range(nfmd.num_freqs):
            # If it's not the lowest-freq mode (assumed to be the mean)
            if i != np.argmin(mean_freqs):
                # Extract IFs, IAs, and A vector for the mode:
                IF = freqs[:,i]
                IA = amps[:,i]
                A = A[:, i::nfmd.num_freqs]
                # Initialize the Mode object:
                mode = NFMDMode(IF, IA, A, t[nfmd.mid_idcs])
                # Store the mode in the 
                self.modes.append(mode)