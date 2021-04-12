import numpy as np
import torch


class NFMD:
    def __init__(self, signal, num_freqs, window_size,
                 windows=None,
                 optimizer=torch.optim.SGD,
                 optimizer_opts={'lr': 1e-4},
                 max_iters=1000,
                 target_loss=1e-4,
                 device='cpu'):
        '''
        Initialize the object

        Parameters
        ----------
        Signal: numpy.ndarray
            temporal signal to be analyzed (should be 1-D)
        num_freqs: integer
            number of frequencies to fit to signal.
            (Note: The 'mean' mode counts as a frequency mode)
        optimizer: optimizer object (torch.optim)
            Optimization algorithm to employ for learning.
        optimizer_opts: dict
            Parameters to pass to the optimizer class.
        max_iters: int
            number of steps for optimizer to take (maximum)
        target_loss: float
            the loss value at which the window is considered sufficiently 'fit'
            (note: setting this too low can cause issues by pushing freqs to 0)
        device: string
            device to use for optimization
            (Note: default 'cpu', but could be 'cuda' with GPU)

        '''
        # Signal -- assumed 1D, needs to be type double
        self.x = signal.astype(np.double).flatten()
        self.n = signal.shape[0]
        # Signal Decomposition options
        self.num_freqs = num_freqs
        self.window_size = window_size
        self.windows = windows
        if not windows:
            self.windows = self.n
        # Stochastic Gradient Descent Options
        self.optimizer = optimizer
        self.optimizer_opts = optimizer_opts
        # If the learning rate is specified, scale it by
        # window size
        if 'lr' in optimizer_opts:
            self.optimizer_opts['lr'] /= window_size
        self.max_iters = max_iters
        self.target_loss = target_loss
        self.device = device


    def decompose_signal(self, update_freq: int = None):
        '''
        Compute the slices of the windows used in the analysis.
        Note: this is equivalent to computing rectangular windows.

        Parameters
        ----------
        update_freq: TYPE integer
            The number of optimizer steps between printed update statements.
        Returns
        -------
        freqs: numpy.ndarray
            frequency vector
        A: numpy.ndarray
            coefficient vector
        losses: numpy.ndarray
            Fit loss (MSE) for each window
        indices: list
            list of slice objects. each slice describes fit window indices.

        '''
        # Compute window indices
        self.compute_window_indices()

        # Determine if printing updates
        verbose = update_freq != None

        # lists for results
        self.freqs = []
        self.A = []
        self.losses = []
        self.window_fits = []  # Save the model fits

        # Tracker variables for previous freqs and A
        prev_freqs = None
        prev_A = None

        # iterate through each window:
        for i, idx_slice in enumerate(self.indices):
            # If update frequency is requested, print an update
            # at window <x>
            if verbose:
                if i % update_freq == 0:
                    print("{}/{}".format(i, len(self.indices)), end="|")

            # Access data slice
            x_i = self.x[idx_slice].copy()

            # Determine number of SGD iterations to allow
            max_iters = self.max_iters

            if i == 0:
                max_iters = 10000

            # Fit data in window to model
            loss, freqs, A = self.fit_window(x_i,
                                             freqs=prev_freqs,
                                             A=prev_A)

            # Store the results
            self.freqs.append(freqs)
            self.A.append(A)
            self.losses.append(loss)

            # Set the previous freqs and A variables
            prev_freqs = freqs
            prev_A = A

        self.freqs = np.asarray(self.freqs)
        self.A = np.asarray(self.A)
        self.losses = np.asarray(self.losses)

        return self.freqs, self.A, self.losses, self.indices


    def compute_window_indices(self):
        '''
        Sets the 'indices' attribute with computed index slices corresponding
        to the windows used in the analysis.
        Note: this is equivalent to computing rectangular windows.

        Parameters
        ----------
        None.
        Returns
        -------
        None.

        '''
        # Define how many points between centerpoint of windows
        increment = int(self.n/self.windows)
        window_size = self.window_size
        # Initialize the indices lists
        self.indices = []
        self.mid_idcs = []
        # Populate the indices lists
        for i in range(self.windows):
            # Compute window slice indices
            idx_start = int(max(0, i*increment-window_size/2))
            idx_end = int(min(self.n, i*increment+window_size/2))
            if idx_end-idx_start == window_size:
                # Add the index slice to the indices list
                self.indices.append(slice(idx_start, idx_end))
                idx_mid = int((idx_end+idx_start)/2)
                self.mid_idcs.append(idx_mid)


    def fit_window(self, xt, freqs=None, A=None):
        '''
        Fits a set of instantaneous frequency and component coefficient vectors
        to the provided data.

        Parameters
        ----------
        xt : TYPE numpy.ndarray
            Temporal data of dimensions [T, ...]
        freqs : TYPE numpy.ndarray, optional
            1D vector of (guess) instantaneous frequencies
            (Note: assumes dt=1 in xt data array)
        A : TYPE numpy.ndarray, optional
            1D vector of cosine/sine coefficients
        max_iters : TYPE int, optional
            Number of optimization steps to take
        Returns
        -------
        loss: float
            the loss for the fit window (mean squared error)
        freqs: numpy.ndarray
            frequency vector of instantaneous frequencies
        A: numpy.ndarray
            coefficient vector of component (sine/cosine) coefficients

        '''
        # If no frequency is provided, generate initial frequency guess:
        if freqs is None:
            freqs, A = self.fft(xt)

        # Then begin SGD
        loss, freqs, A = self.sgd(xt, freqs, A, max_iters=self.max_iters)

        return loss, freqs, A


    def fft(self, xt):
        '''
        Given temporal data xt, fft performs the initial guess of the
        frequencies contained in the data using the FFT.

        Parameters
        ----------
        xt : TYPE: numpy.array
            Temporal data of dimensions [T, ...]

        Returns
        -------
        freqs: numpy.ndarray
            vector of instantaneous frequency estimates for each timepoint
        A: numpy.ndarray
            vector of component coefficients

        '''
        # Ensure input signal is 1D:
        if len(xt.shape) == 1:
            xt = xt.reshape(-1,1)

        # Gather model-fitting parameters
        k = self.num_freqs
        N = xt.shape[0]

        # Initialize a list of frequencies:
        freqs = []

        for i in range(k):

            if len(freqs) == 0:
                residual = xt
            else:
                t = np.expand_dims(np.arange(N)+1, -1)
                ws = np.asarray(freqs)
                Omega = np.concatenate([np.cos(t*2*np.pi*ws),
                                        np.sin(t*2*np.pi*ws)], -1)
                A = np.dot(np.linalg.pinv(Omega), xt)

                pred = np.dot(Omega, A)

                residual = pred-xt

            ffts = 0

            for j in range(xt.shape[1]):
                ffts += np.abs(np.fft.fft(residual[:, j])[:N//2])

            w = np.fft.fftfreq(N, 1)[:N//2]
            idxs = np.argmax(ffts)

            freqs.append(w[idxs])
            ws = np.asarray(freqs)

            t = np.expand_dims(np.arange(N)+1, -1)

            Omega = np.concatenate([np.cos(t*2*np.pi*ws),
                                    np.sin(t*2*np.pi*ws)], -1)

            A = np.dot(np.linalg.pinv(Omega), xt)

        return freqs, A

    def sgd(self, xt, freqs, A, max_iters=None):
        '''
        Given temporal data xt, sgd improves the initial guess of omega
        by SGD. It uses the pseudo-inverse to obtain A.

        Parameters
        ----------
        xt : numpy.ndarray
            Temporal data of dimensions [T, ...]
        freqs : numpy.ndarray
            frequency vector
        A: numpy.ndarray
            Component coefficient vector
        max_iters:
            Number of optimizer steps to take (maximum)

        Returns
        -------
        loss: float
            the loss for the fit window (mean squared error)
        freqs: numpy.ndarray
            frequency vector of instantaneous frequencies
        A: numpy.ndarray
            coefficient vector of component (sine/cosine) coefficients

        '''
        # Set up PyTorch tensors for SGD
        A = torch.tensor(A, requires_grad=False, device=self.device)
        freqs = torch.tensor(np.asarray(freqs), requires_grad=True, device=self.device)
        xt = torch.tensor(xt, requires_grad=False, device=self.device)

        # Set up PyTorch Optimizer
        o2 = self.optimizer([freqs], **self.optimizer_opts)

        # Time indices
        t = torch.unsqueeze(torch.arange(len(xt),
                                         dtype=torch.get_default_dtype(),
                                         device=self.device)+1, -1)

        # Determine how many iterations will be used
        if not max_iters:
            max_iters = self.max_iters

        # SGD to determine solution
        for i in range(max_iters):
            # Compute new model
            Omega = torch.cat([torch.cos(t*2*np.pi*freqs),
                               torch.sin(t*2*np.pi*freqs)], -1)

            A = torch.matmul(torch.pinverse(Omega.data), xt)

            xhat = torch.matmul(Omega, A)

            # Compute Loss function
            loss = torch.mean((xhat-xt)**2)

            # Take a step
            o2.zero_grad()
            loss.backward()
            o2.step()

            # If loss is below fit threshold, end learning
            if loss < self.target_loss:
                break

        # Store the model fit:
        xhat = xhat.cpu().detach().numpy()
        self.window_fits.append(xhat)

        # Prepare the results
        A = A.cpu().detach().numpy()
        freqs = freqs.cpu().detach().numpy()

        return loss, freqs, A

    def predict(self, T):
        ''' Predicts the data from 1 to T.

        Parameters
        ----------
        T : TYPE int
            Prediction horizon (number of timepoints T)

        Returns
        -------
        TYPE numpy.array
            xhat from 0 to T.
        '''
        t = np.expand_dims(np.arange(T)+1, -1)

        for i, idx_slice in enumerate(self.indices):
            local_freqs = self.freqs[i]
        Omega = np.concatenate([np.cos(t*2*np.pi*self.freqs),
                                np.sin(t*2*np.pi*self.freqs)], -1)
        return np.dot(Omega, self.A)

    def correct_frequencies(self, dt):
        '''
        Compute corrected frequency vector that takes into account the
        timestep dt in the signal

        Parameters
        ----------
        dt: float
            The time step between samples in the signal
        Returns
        -------
        corrected_freqs: numpy.ndarray
            Timestamp-corrected frequency vector

        '''
        corrected_freqs = []
        for freq in self.freqs:
            corrected_freqs.append(freq/dt)
        corrected_freqs = np.asarray(corrected_freqs)
        return corrected_freqs

    def compute_amps(self):
        '''
        Compute the 'amplitude' of the Fourier mode.
        Amplitude = sqrt(A_1^2 + A_2^2)

        Parameters
        ----------
        None.
        Returns
        -------
        Amps: numpy.ndarray
            Amplitude vector, length = num_freqs

        '''
        # initialize amps list
        Amps = np.ndarray((self.A.shape[0], self.num_freqs))
        #print(Amps.shape)
        # Populate amps list
        for i, A in enumerate(self.A):
            #print(A.shape)
            # Reshape the As list into a 2 x k matrix of
            # cosine and sine coefficients
            AsBs = A.reshape(-1,self.num_freqs)
            # Compute amplitude of each mode:
            for j in range(AsBs.shape[-1]):
                Amp = complex(*AsBs[:,j])
                Amps[i,j]=abs(Amp)
        Amps = np.asarray(Amps)
        return Amps

    def compute_mean(self, lf_mode=None):
        '''
        Computes the value of the mean mode. The mode is constructed by
        taking the value of the fit mean mode at the center of the window
        for each window data was fit in, and concatenating the center values.

        Parameters
        ----------
        lf_mode: optional, integer
            The index of the mode that is known to represent the mean.
            Note: if not provided, the lowest-average-IF mode is assumed to be
            the mean.
        Returns
        -------
        means: numpy.ndarray
            The reconstructed mean signal at each time point.

        '''
        # Initialize empty array
        means = np.ndarray(len(self.mid_idcs))
        # Identify the low-frequency mode based on initial frequency estimate
        if lf_mode is None:
            lf_mode = np.argmin(np.mean(self.freqs[:,:], axis=0))
        mid_idx = int(self.window_size/2)
        # Iterate through each fourier object and compute the mean
        for i in range(len(self.mid_idcs)):
            # Grab the frequency and the amplitudes
            freq = self.freqs[i, lf_mode]
            A = self.A[i, lf_mode::self.num_freqs]
            # Compute the estimate
            t = np.expand_dims(np.arange(self.window_size)+1, -1)
            Omega = np.concatenate([np.cos(t*2*np.pi*freq),
                                    np.sin(t*2*np.pi*freq)], -1)
            fit = np.dot(Omega, A)
            # Grab the centerpoint and add it to the means list
            means[i] = fit[mid_idx]
        return means

    def predict_window(self, i):
        '''
        Show the sum of the modes fit to a window of index i.

        Parameters
        ----------
        i: integer
            index of the window to retrieve the fit for.
        Returns
        -------
        fit: numpy.ndarray
            The sum of the reconstructed modes for the given window.

        '''
        return self.window_fits[i]