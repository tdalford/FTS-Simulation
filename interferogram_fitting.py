import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
c = 300.


def generate_spectrum(interferogram, fts_step_size):
    # Fourier transform interferogram
    windowed_interferogram = np.hanning(int(np.shape(interferogram)[
        0])) * interferogram
    S = np.fft.rfft(windowed_interferogram)
    fft = np.abs(S)
    fft_freqs = np.fft.rfftfreq(len(interferogram), d=(4 * fts_step_size))
    return fft_freqs, fft


def get_frequency_shift_and_peak_width(initial_freq, interferogram, step_size,
                                       print_vals=False, plot_vals=False):
    # generate FFT
    freqs, fft = generate_spectrum(interferogram - np.mean(
        interferogram), step_size)

    # find the peak of the FFT and compare to the frequency
    peak_index = np.argmax(fft[3:])
    peak = freqs[3:][peak_index] * c
    peak_diff = (peak - initial_freq) / initial_freq

    # Fit a gaussian to the FFT peak and take the peak of that and compare
    popt = gaussian_fit(c * freqs[3:], fft[3:])
    if (popt is None):
        gaussian_diff = peak_diff  # just make it the peak difference
        fwhm = 2 * np.sqrt(2 * np.log(2)) * np.std(fft[3:])
    else:
        gaussian_diff = (popt[1] - initial_freq) / initial_freq
        # The FWHM is simply 2sqrt(log(2)) * sigma
        fwhm = 2 * np.sqrt(2 * np.log(2)) * np.abs(popt[2])

    if (print_vals):
        print('initial frequency = %s' % initial_freq)
        print('maximumum in data obtained at frequency %s' % peak)
        print('measured frequency from gaussian fit = %s' % popt[1])

    if (plot_vals):
        plt.plot(c * freqs, fft, '.', label='data')
        x = np.linspace(0, 315, 2000)
        plt.plot(x, gaussian(x, *popt), alpha=.5, label='fit')
        plt.axvline(x=float(initial_freq), color='green',
                    label=str(initial_freq)+'GHz')
        plt.xlabel('Frequency (GHz)', color='black')
        plt.tick_params(colors='black')
        plt.xlim(initial_freq - 20, initial_freq + 20)
        plt.legend()

    if (print_vals):
        print('FWHM from gaussian fit = %s' % fwhm)

    return peak_diff, gaussian_diff, fwhm


def gaussian(x, A, x0, sigma):
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def gaussian_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    try:
        popt, pcov = curve_fit(gaussian, x, y, p0=[
                               max(y), mean, sigma], maxfev=10000)
        return popt
    except RuntimeError:
        return None


def get_amplitude(interferogram, step_size, print_vals=False,
                  plot_vals=False):
    # get maximum spectrum value (find the peak of the FFT)
    freqs, fft = generate_spectrum(interferogram, step_size)
    max_val = np.max(fft[3:])

    # fit this to a gaussian and find the maximum value
    popt = gaussian_fit(c * freqs[3:], fft[3:])
    if (popt is None): # just take the max value of the data as an approx
        gaussian_max_val = max_val
    else:
        gaussian_max_val = popt[0]

    if (print_vals):
        print('maximum value of amplitude = %s' % max_val)
        print('maximum value of gaussian = %s' % gaussian_max_val)

    if (plot_vals):
        plt.plot(c * freqs[3:], fft[3:], '.', label='data')
        x = np.linspace(0, 1000, 5000)
        plt.plot(x, gaussian(x, *popt), alpha=.5, label='fit')
        plt.xlabel('Frequency (GHz)', color='black')
        plt.tick_params(colors='black')
        max_freq = np.argmax(fft[3:])
        plt.xlim(max_freq - 20, max_freq + 20)
        plt.legend()

    return gaussian_max_val, max_val
