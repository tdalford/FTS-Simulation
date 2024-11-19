from fts_coupling_optics_geo import *
import fts_coupling_optics_geo as fts
import coupling_optics_sims as csims
#from RayTraceFunctionsv2 import *
from scipy.optimize import curve_fit
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager, Semaphore
import RayTraceFunctionsv2 as rt

c = 300.
LAST_LENS_EDGE = [-231.24377979, -266.21940725, 0.]
COUPLING_OPTICS_ORIGIN = [-233.28894593160666, -276.84436350628596, 0.]


def get_aspect(config, aspect, element, number):
    '''Obtain the config for a given aspect, element, and number

    Parameters:
        config (yaml file)  -- yaml configuration file loaded
        aspect (str)        -- aspect of the FTS (origins, angles, coefficients, etc)
        element (str)       -- element of the FTS for which this aspect is defined 
                              (ellipses, mirror, polarizers)
        number (int)        -- number of the element that we're specifically interested
                            in (1-10 for ellipses, 1-4 for polarizers)'''

    def get_item(dic, key, num):
        if type(dic[key]) is dict:
            return dic[key][num]
        else:
            return dic[key]

    if element is None:
        return get_item(config, aspect, number)
    else:
        return get_item(config[aspect], element, number)


def step_rays(starting_rays, config, ray_func, *ray_func_args, final_dist=50,
              debug=1):
    # Data structure which contains starting point, vector, length for each ray
    # through the sim
    total_ray_points = []
    total_ray_vectors = []
    total_ray_distances = []
    counts = []
    max_count = None

    for starting_ray in starting_rays:
        current_rays = ray_func(
            starting_ray, *ray_func_args, return_all_rays=True)
        # We want to save the point, vector, and distance travelled for each of
        # these rays!
        points = []
        vectors = []
        distances = [starting_ray[4]]
        count = 0
        max_count = len(current_rays) + 1
        for i, ray in enumerate(current_rays):
            if ray is not None:
                count += 1
                points.append(ray[2])
                vectors.append(ray[3])

                # If we're not at the final ray and the next ray hit,
                # calculated the distance!
                if (i < len(current_rays) - 1):
                    if current_rays[i + 1] is not None:
                        distances.append(
                            current_rays[i + 1][4] - current_rays[i][4])
                else:
                    # check and see if the final ray hit the detector!
                    final_ray = rt.get_final_rays_tilt(
                        [ray], config['detector']['center'],
                        config['detector']['range'],
                        config['detector']['normal_vec'])
                    if (final_ray) != []:
                        distances.append(rt.dist(ray[2], final_ray[0][2]))
                        count += 1

                    else:
                        distances.append(final_dist)
            else:
                # The ray did not make it to the end.
                # Append a final distance for the last ray so we can see where
                # it went.
                distances.append(final_dist)

        total_ray_points.append(points)
        total_ray_vectors.append(vectors)
        total_ray_distances.append(distances)
        counts.append(count)

    assert debug in [0, 1, 2]
    if (debug == 2):
        print('final ray counts = %s' % counts)
    if (debug == 1):
        print('initial number of rays = %s' % len(starting_rays))
        print('total number of rays making past the first ellipse = %s'
              % np.sum(np.array(counts) != 1))
        print('total number of rays making it all the way through = %s'
              % np.sum(np.array(counts) == max_count))
    return total_ray_points, total_ray_vectors, total_ray_distances


def plot_interferogram(delay, ij, ax, labels=True):
    ax.plot(delay, ij, linewidth=.8)
    if (labels):
        ax.set_xlabel('Optical Delay (mm)', color='black')

    ax.tick_params(colors='black')
    return


def plot_fft(initial_freq, freqs, fft, ax, zoom=True, zoom_amt=20,
             labels=True):
    ax.plot(c * freqs[3:], fft[3:])
    if (labels):
        ax.set_xlabel('Frequency (GHz)', color='black')

    ax.tick_params(colors='black')
    ax.axvline(x=float(initial_freq), color='green',
               label=str(initial_freq) + ' GHz')
    if (zoom):
        ax.set_xlim(initial_freq - zoom_amt, initial_freq + zoom_amt)
    ax.legend()
    return


def plot_spectra(initial_freq, delay, ij, zoom=True, zoom_amt=20, ymax=18,
                 fac=.95):
    # generate spectrum
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    freqs, fft = generate_spectrum(initial_freq, ij, ymax=ymax, fac=fac)

    plot_interferogram(delay, ij, axes[0])

    # plot spectrum with expected frequency value
    plot_fft(initial_freq, freqs, fft, axes[1], zoom=zoom,
             zoom_amt=zoom_amt)


def generate_spectrum(initial_freq, ij, ymax=18, fac=.95):
    # Fourier transform interferogram
    d1 = ij
    D = np.hanning(int(np.shape(d1)[0])) * d1
    S = np.fft.rfft(D)
    fft = np.abs(S)
    xtot = 2 * ymax * fac * 4  # total length of the scan

    fft_freqs = np.fft.rfftfreq(len(d1), d=(xtot / len(d1)))
    return fft_freqs, fft  # - fft_freqs[1] / 2, fft


def plot_freq_interferogram(test_freq, rays, delay, c=300., ymax=32, fac=.95):
    ij = []
    for y_rays in rays:
        ij.append(rt.sum_power_vectorized(y_rays, c / test_freq))

    plot_spectra(test_freq, delay, ij, ymax=ymax, fac=fac)


def scan_frequency_range(delay, outrays, freqs, ymax):
    gaussian_amplitudes = []
    fft_amplitudes = []
    peak_shifts = []
    gaussian_shifts = []
    peak_widths = []

    for freq in freqs:
        ij = rt.get_interferogram(outrays, (c / freq))
        peak_shift, gaussian_shift, peak_width = \
            get_frequency_shift_and_peak_width(freq, ij, ymax, plot_vals=False,
                                               print_vals=False)
        gaussian_amplitude, max_fft = get_amplitude(
            freq, ij, ymax, print_vals=False)

        peak_shifts.append(peak_shift)
        gaussian_shifts.append(gaussian_shift)
        peak_widths.append(peak_width)
        gaussian_amplitudes.append(gaussian_amplitude)
        fft_amplitudes.append(max_fft)

    return peak_shifts, gaussian_shifts, peak_widths, gaussian_amplitudes, \
        fft_amplitudes


def get_frequency_shift_and_peak_width(initial_freq, ij, ymax,
                                       print_vals=True, plot_vals=True):
    # generate FFT
    freqs, fft = generate_spectrum(initial_freq, ij, ymax=ymax, fac=1)

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
        plt.plot(c * freqs[3:], fft[3:], '.', label='data')
        x = np.linspace(0, 1000, 5000)
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


def get_amplitude(initial_freq, ij, ymax, print_vals=True, plot_vals=False):
    # get maximum spectrum value (find the peak of the FFT)
    freqs, fft = generate_spectrum(initial_freq, ij, ymax=ymax, fac=1)
    max_val = np.max(fft[3:])

    # fit this to a gaussian and find the maximum value
    popt = gaussian_fit(c * freqs[3:], fft[3:])
    if (popt is None):
        gaussian_max_val = max_val
    else:
        #x = np.linspace(0, 1000, 5000)
        #gaussian_max_val = np.max(gaussian(x, *popt))
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
        plt.xlim(initial_freq - 20, initial_freq + 20)
        plt.legend()

    return gaussian_max_val, max_val


def rays_to_matrix(rays):
    max_rays_num = max([len(y_rays) for y_rays in rays])
    # get max number and pad to that these fit nicely into a matrix
    total_rays = []
    for y_rays in rays:
        # only keep polarization, intensity, and phase
        # could also keep detector position and angle separately
        # if we're curious!
        power_vals_mask = [0, 1, 4]
        power_vals = np.array(np.array(y_rays, dtype='object')[
            :, power_vals_mask], dtype='float64')
        # pad rows now
        total_vals = np.zeros((max_rays_num, 3))
        total_vals[:power_vals.shape[0]] = power_vals
        total_rays.append(total_vals)

    return np.array(total_rays)


def plot_shifts_end(shift, freqs, gaussian_shifts, gaussian_amplitudes,
                    figsize=(12, 5)):
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    # from gaussian fitted maximum FFT value')
    title = ('(%.2g, %.2g, %.2g) mm. det shift from focus' % (
        tuple(np.array(shift) * 25.4)))
    ax[0].set_title('%s \n, frequency shift' % title)
    ax[0].plot(freqs, np.array(gaussian_shifts), '.', color='green')
    ax[0].set_xlabel('frequency [Ghz]')
    ax[0].set_ylabel('fractional frequency shift [Ghz]')
    # ax[0].set_ylim(-.1, 0)
    ax[0].grid()

    # from gaussian fitted maximum FFT value')
    ax[1].set_title('%s \n, output amplitude over frequency' % title)
    ax[1].plot(freqs, np.array(gaussian_amplitudes), '.', color='blue')
    ax[1].set_xlabel('frequency [Ghz]')
    ax[1].set_ylabel('FTS normalized amplitude')
    ax[1].grid()

    # plt.suptitle('%s mm. source shifting' % (np.array(shift) * 25.4))
    plt.tight_layout()
    # plt.show()


def plot_shifts(delay, ray_mat, freqs, ymax, shift=None, plot=False):
    peak_shifts, gaussian_shifts, peak_widths, gaussian_amplitudes, \
        fft_maxes = scan_frequency_range(delay, ray_mat, freqs, ymax)

    if plot:
        plot_shifts_end(shift, freqs, gaussian_shifts, gaussian_amplitudes)
    return gaussian_shifts, peak_shifts, gaussian_amplitudes, fft_maxes


def get_shifts(rays, config, n_mirror_positions, possible_paths, ymax,
               shift=None, debug=False):
    delay, final_rays = rt.run_all_rays_through_sim_optimized(
        rays, config, n_mirror_positions, paths=possible_paths, ymax=ymax,
        progressbar=debug)
    ray_mat = rays_to_matrix(final_rays)
    nyquist_freq = c / ((2 * ymax / n_mirror_positions) * 4 * 2)
    freqs = np.arange(15, nyquist_freq - 10 + 1)
    gaussian_shifts, peak_shifts, gaussian_amplitudes, fft_maxes = plot_shifts(
        delay, ray_mat, freqs, ymax, shift=shift)

    return delay, ray_mat, gaussian_shifts, peak_shifts, \
        gaussian_amplitudes, fft_maxes, freqs


def smart_rms(timeseries, n_iters, threshold):
    ok = np.where(timeseries == timeseries)  # CAN PROBABLY TAKE THIS OUT
    timeseries = timeseries[ok]
    for _ in range(n_iters):
        rms_tmp = np.std(timeseries)
        mean_tmp = np.mean(timeseries)
        if rms_tmp == 0:  # don't divide or else we'll all be nans...
            return (mean_tmp, rms_tmp)
        good = np.where(np.abs(timeseries - mean_tmp) / rms_tmp < threshold)
        timeseries = timeseries[good]
    return(mean_tmp, rms_tmp)


def transform_rays_to_fts_frame(rays):
    source_point_origin = LAST_LENS_EDGE
    angle = .190161
    new_rays = []
    for ray in rays:
        new_ray = [ray[0], ray[1], None, None, ray[4]]
        # .19635 #.253406]) #should actually be 10.89 deg #11.25 degrees now
        new_vec = rt.rotate(ray[3], [0, 0, angle])
        new_ray[2] = np.add(rt.rotate(ray[2], [0, 0, angle]),
                            source_point_origin)
        new_ray[3] = new_vec
        new_rays.append(new_ray)

    return new_rays


def add_shift_attrs(total_attrs, shift, n_linear, n_mirror_positions,
                    possible_paths, y_max, config, semaphore, debug):
    start_rays = csims.get_final_rays_reversed(
        shift, theta_bound=.25, n_linear=n_linear, y_ap=-.426)
    transformed_start_rays = transform_rays_to_fts_frame(start_rays)

    delay, ray_mat, gaussian_shifts, peak_shifts, gaussian_amplitudes, \
        fft_maxes, freqs = get_shifts(
            transformed_start_rays, config, n_mirror_positions, possible_paths,
            y_max, shift=shift, debug=debug)

    mean_freq, std_freq = smart_rms(np.array(gaussian_shifts), 4, 4)
    # just take the normal RMS here
    mean_amp, std_amp = smart_rms(np.array(gaussian_amplitudes), 1, 10)
    total_attrs.put([shift, delay, gaussian_shifts, peak_shifts,
                     gaussian_amplitudes, fft_maxes, mean_freq, std_freq,
                     mean_amp, std_amp, freqs, ray_mat])
    semaphore.release()
    return


def get_freq_shifts_threaded(
        x_vals, y_vals, z_vals, n_linear, n_mirror_positions, possible_paths,
        y_max, config, debug=False):
    processes = []
    max_processes = 55
    semaphore = Semaphore(max_processes)
    manager = Manager()
    total_attrs = manager.Queue()

    for x in x_vals:
        for y in y_vals:
            for z in z_vals:
                shift = [x, y, z]
                semaphore.acquire()
                process = Process(target=add_shift_attrs, args=(
                    total_attrs, shift, n_linear, n_mirror_positions,
                    possible_paths, y_max, config, semaphore, debug),
                    daemon=True)

                print('process %s starting.' % len(processes))
                process.start()
                processes.append(process)

    # Wait for all the processes to finish before converting to a list.
    for i, process in enumerate(processes):
        print('process %s joining.' % i)
        process.join()
        print('process %s finishg.' % i)

    # convert the shared queue to a list.
    attr_list = []
    while (total_attrs.qsize() != 0):
        attr_list.append(total_attrs.get())
    return attr_list


def main():
    # FTS_stage_step_size = 0.375  # FTS step size in mm
    # FTS_stage_step_size = 0.15  # FTS step size in mm
    # FTS_stage_throw = 35.     # total throw extent in mm

    # n_mirror_positions = (2 * FTS_stage_throw / FTS_stage_step_size)
    # n_mirror_positions = 2 * (FTS_stage_throw / .15)

    FTS_stage_throw = 20.     # total throw extent in mm
    FTS_stage_step_size = 0.1  # FTS step size in mm
    # n_mirror_positions = (20 * 2 / .1)
    n_mirror_positions = (2 * FTS_stage_throw / FTS_stage_step_size)

    with open("lab_fts_dims_act.yml", "r") as stream:
        config = yaml.safe_load(stream)
        # set the detector (really 'source') range to the desired amount
        # config['detector']['range'] = 30.

    print('center = %s, range = %s' % (
        config['detector']['center'], config['detector']['range']))
    possible_paths = rt.get_possible_paths()

    # x_vals = np.linspace(-.65, .65, 1)
    # y_vals = np.linspace(-.65, .65, 1)
    # z_vals = np.linspace(0, 0, 1)

    x_vals = np.linspace(0, 0, 1)
    y_vals = np.linspace(0, 0, 1)
    z_vals = np.linspace(0, 0, 1)

    # x_vals = np.linspace(0, 0, 1)
    # y_vals = np.linspace(0, 0, 1)
    # cm to inches
    # z_vals = np.array([-4, -2.5, -2, -1, 1, 2, 2.5, 4]) / (2.54)  # cm to in

    # x_vals = np.linspace(-.2, -.2, 1)
    # y_vals = np.linspace(.2, .2, 1)
    # z_vals = np.linspace(-2, -2, 1)
    total_attrs_xy = get_freq_shifts_threaded(
        x_vals, y_vals, z_vals, 10, n_mirror_positions, possible_paths,
        FTS_stage_throw, config, debug=True)

    # pickle.dump(total_attrs_y, open(
    #     "total_attrs_z_shift_1_5_10_25_35_range_42.p", "wb"))

    # pickle.dump(total_attrs_y, open("z_2_one_point_50_35.p", "wb"))

    pickle.dump(total_attrs_xy, open(
        "data/total_attrs_xz_15_15_20_57.p", "wb"))

    # a run with only y shifts here!

    # save this for loading elsewhere
    print('finished!')


if __name__ == '__main__':
    main()
    exit()
