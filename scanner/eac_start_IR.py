import numpy as np
from scipy.signal import hilbert, correlate

class itaAudio:
    def __init__(self, timeData, nSamples, nChannels):
        self.timeData = timeData
        self.nSamples = nSamples
        self.nChannels = nChannels

def ita_start_IR(audioObj, threshold=20, correlation=False, order=2):
    if isinstance(threshold, str):
        threshold = float(threshold)
        if np.isnan(threshold):
            raise ValueError('threshold is NaN!')

    if not correlation:
        sampleStart = local_search_ISO3382(audioObj, threshold)
    else:
        sampleStart = local_search_minPhase_correlation(audioObj, order)
    return sampleStart

def local_search_ISO3382(input, threshold):
    IRsquare = input.timeData**2

    # Assume the last 10% of the IR is noise, and calculate its noise level
    NoiseLevel = np.mean(IRsquare[round(0.9*len(IRsquare)):, :], axis=0)

    # Get the maximum of the signal, that is the assumed IR peak
    max_val = np.max(IRsquare, axis=0)
    max_idx = np.argmax(IRsquare, axis=0)

    # Check if the SNR is enough to assume that the signal is an IR
    idxNoShift = (max_val < 100 * NoiseLevel) | (max_idx > round(0.9 * input.nSamples))
    sampleStart = np.ones_like(max_val)

    for idx in range(input.nChannels):
        if idxNoShift[idx]:
            continue
        
        if max_idx[idx] > 1:
            abs_dat = 10 * np.log10(IRsquare[:max_idx[idx], idx]) - 10 * np.log10(max_val[idx])
            lastBelowThreshold = np.where(abs_dat < -abs(threshold))[0]
            
            if len(lastBelowThreshold) > 0:
                sampleStart[idx] = lastBelowThreshold[-1]
            else:
                sampleStart[idx] = 1
            
            idx6dBaboveThreshold = np.where(abs_dat[:sampleStart[idx]] > -abs(threshold) + 6)[0]
            
            if len(idx6dBaboveThreshold) > 0:
                tmp = np.where(abs_dat[:idx6dBaboveThreshold[0]] < -abs(threshold))[0]
                if len(tmp) == 0:
                    sampleStart[idx] = 1
                else:
                    sampleStart[idx] = tmp[-1]
    return sampleStart

def local_search_minPhase_correlation(input, order):
    sampleStart = np.ones(input.nChannels)

    for iChannel in range(input.nChannels):
        A = input.timeData[:, iChannel]
        B = ita_minimumphase(A)
        
        X = correlate(A, B, mode='full')
        Y = hilbert(X)
        ind = np.argmax(np.abs(Y))
        y = np.imag(Y[ind - (order//2): ind + (order//2) + 1])
        p = np.polyfit(np.arange(-(order//2), (order//2) + 1), y, order)
        r = np.roots(p)
        
        if np.all(np.isreal(r)):
            r = r[np.argmin(np.abs(r))]
            sampleStart[iChannel] = ind + r - input.nSamples
        else:
            sampleStart[iChannel] = 0
    return sampleStart

def ita_minimumphase(signal):
    # Convert to minimum phase (placeholder implementation)
    return np.real(np.fft.ifft(np.abs(np.fft.fft(signal))))

# Example usage
# timeData should be a 2D numpy array with shape (nSamples, nChannels)
# audioObj = itaAudio(timeData, nSamples, nChannels)
# sampleStart = ita_start_IR(audioObj, threshold=20, correlation=False, order=2)
