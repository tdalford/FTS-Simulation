from fts_coupling_optics_geo import *
import fts_coupling_optics_geo as fts
import coupling_optics_sims as csims
import yaml
import pickle
import numpy as np
from multiprocessing import Process, Manager, Semaphore
import RayTraceFunctions as rt
from tqdm import tqdm
import time

c = 300.
LAST_LENS_EDGE = [-231.24377979, -266.21940725, 0.]
COUPLING_OPTICS_ORIGIN = [-233.28894593160666, -276.84436350628596, 0.]
IN_TO_MM = 1 / (csims.mm_to_in)
# DET_SIZE = 5  # in mm
# For LF: 14mm, for MF/UHF: 5mm
FTS_BEAM_ANGLE = -0.190161


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


def remove_outliers(out, threshold=5, debug=False):
    new_out = []
    mean_x = csims.FOCUS[0]
    mean_y = csims.FOCUS[1]
    if (debug):
        print(f'focus is at {np.array(csims.FOCUS)[[0, 1]]}')

    for i in range(out.shape[1]):
        point = out[[0, 1, 2], i]
        if np.abs(point[0] - mean_x) > threshold or np.abs(
                point[1] - mean_y) > threshold:
            if (debug):
                print(f'Taking away outlier point {point}')
            continue
        new_out.append(out[:, i])
    return np.array(new_out).T


def gaussian_sigma_from_fwhm(fwhm_rad):
    return fwhm_rad / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def weight_rays_gaussian(rays, normal_vector, fwhm_deg=30):
    # Currently approximates the feedhorn ray distribution to the distribution
    # to a radially symmetric gaussian with FWHM around 30 degrees. This is a
    # good enough approximation for our purposes, although in actuality the
    # FWHM can vary from ~15 to 60 degrees, depending on the band, and the x
    # and y FWHMs vary slightly.
    cos_angle = rays[[3, 4, 5]].T @ normal_vector
    theta = np.arccos(cos_angle)
    sigma = gaussian_sigma_from_fwhm(np.deg2rad(fwhm_deg))
    weight = np.exp(-0.5 * (theta / sigma**2))
    # Multiply the intensity by our weight.
    rays[7] *= weight
    return rays

# just subtract our z coordinate until we hit -20.9
def get_distances_at_z(out, z_coordinate):
    # z position is out[2], z direction is out[5]
    iters = (out[2] - z_coordinate) / out[5]
    return out[8] - iters / csims.mm_to_in


def get_rays_at_z(out, z_coordinate):
    iters = (out[2] - z_coordinate) / out[5]
    new_points = out[[0, 1, 2]] - out[[3, 4, 5]] * iters

    new_out = out.copy()
    new_out[[0, 1, 2]] = new_points
    new_out[8] = get_distances_at_z(out, z_coordinate)
    return new_out


def get_rotation_matrix(a, b):
    v = np.cross(a, b)
    s = np.sqrt(np.sum(np.square(v)))
    c = np.dot(a, b)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.identity(3) + vx + np.linalg.matrix_power(vx, 2) * (1 - c) / (
        s ** 2)


def transform_points(x_vals, y_vals, z_vals, new_origin, rotation_matrix):
    XTR = []
    YTR = []
    ZTR = []
    for i in range(0, len(x_vals)):
        v = [x_vals[i], y_vals[i], z_vals[i]]
        # v2R = rotate(v, thetaxyz)
        v2R = np.dot(v, rotation_matrix)
        v2RS = v2R + new_origin
        XTR.append(v2RS[0])
        YTR.append(v2RS[1])
        ZTR.append(v2RS[2])
    return XTR, YTR, ZTR


def transform_rays_to_coupling_optics_frame(rays):
    # we want the rays essentially directly after they hit the last polarizer
    # in the FTS and then we need to calculate the distance between this
    # polarizer and the first lens of the coupling optics

    # first we need to transform the rays' points and normal vectors to the
    # frame of the coupling optics

    # Here we really need to make sure we're properly rotating this ray really
    # the ray should stop at the plane which makes the same beam angle as the
    # coupling optics actually things should be fine I think, but just in case
    # do it this way I guess
    # don't stop at (0, 0, 0), stop at (0, -.426, 0) equivilently
    coupling_optics_origin = COUPLING_OPTICS_ORIGIN
    factor = csims.mm_to_in
    new_rays = []
    for ray in rays:
        new_ray = [ray[0], ray[1], None, None, ray[4]]
        # switch the x and z coordinate of these!
        new_vec = rt.rotate(ray[3], [0, 0, FTS_BEAM_ANGLE])
        new_ray[2] = factor * np.flip(rt.rotate(np.subtract(
            ray[2], coupling_optics_origin), [0, 0, FTS_BEAM_ANGLE]) * [
                1, -1, 1])
        new_ray[3] = np.flip(csims.normalize(factor * new_vec * [1, -1, 1]))
        new_rays.append(new_ray)

    return new_rays


def create_source_rays_uniform_from_start_displacement(
        source_origin, source_normal_vec, horiz_displacement,
        vert_displacement, n_linear_theta, n_linear_phi, config,
        check_rays=True, theta_bound=np.pi / 2, timeout=10):
    # first create rays distributed in the upwards cone
    # and then rotate them to center them around the normal
    # also create them around a variety of starting points
    # assume radially symmetric source
    rotation_matrix = get_rotation_matrix(source_normal_vec, [0, 0, 1])
    rays = []

    # n^2 computations here
    starting_time = time.time()
    for theta_val in np.linspace(0, theta_bound, n_linear_theta):
        for phi_val in np.linspace(0, 2 * np.pi, n_linear_phi):
            if time.time() - starting_time > timeout:
                print('timing out..')
                return rays

            point_origin = [horiz_displacement, vert_displacement, 0]

            # Direction of ray away from the starting point
            r_hat = [np.sin(theta_val) * np.cos(phi_val),
                     np.sin(theta_val) * np.sin(phi_val), np.cos(theta_val)]

            transformed_starting_vector = -1 * np.array(transform_points(
                [r_hat[0]], [r_hat[1]], [r_hat[2]], [0, 0, 0],
                rotation_matrix)).flatten()

            transformed_starting_point = np.array(transform_points(
                [point_origin[0]], [point_origin[1]], [point_origin[2]],
                source_origin, rotation_matrix)).flatten()

            # strategically choose our starting rays such that they make it
            # through the to the first ellipse that we hit
            polarization_angle = .123
            intensity = 1.0
            ray = [polarization_angle, intensity,
                   transformed_starting_point.tolist(),
                   transformed_starting_vector.tolist(), 0]
            # paths = ['OM2', 'A1', 'OM1', 'T4', 'E6']
            if (check_rays):
                paths = ['T4', 'E6']
                final_ray = rt.run_ray_through_sim(ray, config, None, paths)
                if (final_ray is not None):
                    rays.append(ray)
            else:
                rays.append(ray)

    return rays


def create_source_rays_lambertian(
        source_origin, source_normal_vec, horiz_displacement,
        vert_displacement, n_linear_theta, n_linear_phi, config,
        check_rays=True, theta_bound=np.pi / 2, timeout=10,
        count_thetas=False):
    # first create rays distributed in the upwards cone
    # and then rotate them to center them around the normal
    # also create them around a variety of starting points
    # assume radially symmetric source
    rotation_matrix = get_rotation_matrix(source_normal_vec, [0, 0, 1])
    rays = []
    
    if (count_thetas):
        good_vals = []
    
    # n^2 computations here
    starting_time = time.time()
    mu_min = np.cos(theta_bound)
    u_vals = (np.arange(n_linear_theta) + 0.5) / n_linear_theta  # in (0,1)

    # correct truncated Lambert mapping
    mu = np.sqrt(mu_min**2 + (1.0 - mu_min**2) * u_vals)
    theta_vals = np.arccos(mu)
    for theta_val in theta_vals:
        for phi_val in np.linspace(0, 2 * np.pi, n_linear_phi, endpoint=False):
            if time.time() - starting_time > timeout:
                print('timing out..')
                return rays

            point_origin = [horiz_displacement, vert_displacement, 0]

            # Direction of ray away from the starting point
            r_hat = [np.sin(theta_val) * np.cos(phi_val), 
                     np.sin(theta_val) * np.sin(phi_val), np.cos(theta_val)]

            transformed_starting_vector = -1 * np.array(transform_points(
                [r_hat[0]], [r_hat[1]], [r_hat[2]], [0, 0, 0], rotation_matrix)).flatten()

            transformed_starting_point = np.array(transform_points(
                [point_origin[0]], [point_origin[1]], [point_origin[2]], 
                source_origin, rotation_matrix)).flatten()

            
            # polarization vector arbitratily defined here, but strategically
            # not set to some value so that some light has to be reflected and
            # transmitted at the first polarizer.
            polarization_angle = .123
            intensity = 1.0
            ray = [polarization_angle, intensity, transformed_starting_point.tolist(), 
                transformed_starting_vector.tolist(), 0]
            
            if (check_rays):
                # strategically choose our starting rays such that they make it
                # through the to the first ellipse that we hit
                paths = ['T4', 'E6']
                final_ray = rt.run_ray_through_sim(ray, config, None, paths)
                if (final_ray is not None):
                    rays.append(ray)
                    if (count_thetas):
                        good_vals.append([theta_val, phi_val])
            else:
                rays.append(ray)
                if (count_thetas):
                    # for debugging purposes
                    good_vals.append([theta_val, phi_val])
    
    if (count_thetas):
        return rays, good_vals
    return rays

def create_source_rays_lambertian_monte_carlo(
        source_origin, source_normal_vec, horiz_displacement,
        vert_displacement, n_rays, config,
        check_rays=True, theta_bound=np.pi / 2,
        count_thetas=False):
    # first create rays distributed in the upwards cone
    # and then rotate them to center them around the normal
    # also create them around a variety of starting points
    # assume radially symmetric source
    rotation_matrix = get_rotation_matrix(source_normal_vec, [0, 0, 1])
    rays = []
    if (count_thetas):
        good_vals = []
    mu_min = np.cos(theta_bound)

    while len(rays) < n_rays:
        u = np.random.random()
        mu = np.sqrt(mu_min**2 + (1 - mu_min**2) * u)
        theta_val = np.arccos(mu)
        phi_val = np.random.uniform(0, 2 * np.pi)

        point_origin = [horiz_displacement, vert_displacement, 0]

        # Direction of ray away from the starting point
        r_hat = [np.sin(theta_val) * np.cos(phi_val), 
                    np.sin(theta_val) * np.sin(phi_val), np.cos(theta_val)]

        transformed_starting_vector = -1 * np.array(transform_points(
            [r_hat[0]], [r_hat[1]], [r_hat[2]], [0, 0, 0], rotation_matrix)).flatten()

        transformed_starting_point = np.array(transform_points(
            [point_origin[0]], [point_origin[1]], [point_origin[2]], 
            source_origin, rotation_matrix)).flatten()


        # polarization vector arbitratily defined here, but strategically
        # not set to some value so that some light has to be reflected and
        # transmitted at the first polarizer.
        polarization_angle = .123
        intensity = 1.0
        ray = [polarization_angle, intensity, transformed_starting_point.tolist(), 
            transformed_starting_vector.tolist(), 0]

        if (check_rays):
            # strategically choose our starting rays such that they make it
            # through the to the first ellipse that we hit
            paths = ['T4', 'E6']
            final_ray = rt.run_ray_through_sim(ray, config, None, paths)
            if (final_ray is not None):
                rays.append(ray)
                if (count_thetas):
                    good_vals.append([theta_val, phi_val])
        else:
            rays.append(ray)
            if (count_thetas):
                # for debugging purposes
                good_vals.append([theta_val, phi_val])
    
    if (count_thetas):
        return rays, good_vals
    return rays


def get_centers(n_linear_det, det_spacing):
    displacements = []
    centers = np.linspace(-(n_linear_det // 2), (n_linear_det // 2),
                          n_linear_det) * det_spacing
    for x_center in centers:
        for y_center in centers:
            displacements.append((x_center, y_center))
    return displacements


def segment_detector(outrays, n_linear_det, det_spacing, det_size):
    points = outrays[[0, 1, 2]] * IN_TO_MM

    # divide these into segments of length det_spacing
    centers = np.linspace(-(n_linear_det // 2), (n_linear_det // 2),
                          n_linear_det) * det_spacing
    out_data = []
    point_data = []
    for x_center in centers:
        for y_center in centers:
            center = (x_center, y_center + csims.FOCUS[1] * IN_TO_MM)
            # get all the rays that go into this detector
            distance_from_x = np.square(points[0] - center[0])
            distance_from_y = np.square(points[1] - center[1])
            dist_from_center = np.sqrt(distance_from_x + distance_from_y)
            # only take rays that are within the feedhorn outer radius (det
            # size is defined as the diameter)
            segment_within = np.where(dist_from_center < (det_size / 2))[0]
            out_within = outrays[:, segment_within]

            out_data.append(out_within)
            point_data.append([x_center, y_center])
    return out_data, np.array(point_data)


def trace_rays_uniform(start_displacement, n_mirror_positions, ymax,
                       n_linear_theta=100, n_linear_phi=100, debug=False):
    # For each starting position, trace the rays and segment the detectors.
    # Create an interferogram for each detector for each starting position
    # i.e. each starting position should have 36 interferograms (1 raytrace
    # iter).
    with open("lab_fts_dims_act.yml", "r") as stream:
        config = yaml.safe_load(stream)

    starting_rays = create_source_rays_uniform_from_start_displacement(
        config['detector']['center'], config['detector']['normal_vec'],
        start_displacement[0], start_displacement[1], n_linear_theta,
        n_linear_phi, config, theta_bound=np.pi / 2, check_rays=True,
        timeout=100)

    with open("lab_fts_dims_mcmahon_backwards.yml", "r") as stream:
        config = yaml.safe_load(stream)

    possible_paths = [path[::-1] for path in rt.get_possible_paths()]
    delay, final_rays = rt.run_all_rays_through_sim_optimized(
        starting_rays, config, n_mirror_positions, paths=possible_paths,
        ymax=ymax, progressbar=(debug))

    total_outrays = []

    for rays in tqdm(final_rays, disable=(not debug)):
        transformed_rays = transform_rays_to_coupling_optics_frame(rays)
        out_forwards = csims.run_rays_forwards_input_rays(
            transformed_rays, z_ap=csims.FOCUS[2], plot=False)
        total_outrays.append(out_forwards)

    return total_outrays


# Use lambertian distribution.
def trace_rays(start_displacement, n_mirror_positions, ymax, n_rays,
               debug=False):
    # The way I originally defined the FTS in my config files was backwards
    # since Jeff initially wanted to do a reverse raytrace. Later I realized
    # that the nature of the thermal source required a forwards raytrace within
    # the limited sampling space we have. Thus I just grab the source position
    # from what I has originally defined as the 'detector', while the rest of
    # the FTS raytrace is defined via the forward-raytrace oriented config file
    # defined in the aptly named 'lab_fts_dims_mcmahon_backwards.yml'
    with open("lab_fts_dims_act.yml", "r") as stream:
        config = yaml.safe_load(stream)

    starting_rays = create_source_rays_lambertian_monte_carlo(
        config['detector']['center'], config['detector']['normal_vec'],
        start_displacement[0], start_displacement[1], n_rays,
        config, theta_bound=np.pi / 2, check_rays=True)

    with open("lab_fts_dims_mcmahon_backwards.yml", "r") as stream:
        config = yaml.safe_load(stream)

    # I originally described the possible paths through the FTS in reverse
    # also. Realistically I believe they should be symmetric for both forwards
    # and reverse, but originally I had some other operations in there which is
    # why I reverse them.
    possible_paths = [path[::-1] for path in rt.get_possible_paths()]

    # Propagate the source rays through the simulation (to the end of the FTS)
    # for all polarizer paths and central mirror positions.
    delay, final_rays = rt.run_all_rays_through_sim_optimized(
        starting_rays, config, n_mirror_positions, paths=possible_paths,
        ymax=ymax, progressbar=(debug))

    total_outrays = []

    # Transform the rays to the frame of the coupling optics and propgate them
    # through this optical system.
    for rays in tqdm(final_rays, disable=(not debug)):
        transformed_rays = transform_rays_to_coupling_optics_frame(rays)
        out_forwards = csims.run_rays_forwards_input_rays(
            transformed_rays, z_ap=csims.FOCUS[2], plot=False)
        total_outrays.append(out_forwards)

    return total_outrays


# stop outlier removal since it takes too long
def segment_rays(total_out, n_linear_det, det_spacing, det_size):
    total_out_segments = []
    for out in total_out:
        out_segments, det_points = segment_detector(
            out, n_linear_det, det_spacing, det_size)
        total_out_segments.append(out_segments)
    return total_out_segments, det_points


def data_to_matrix(center_data, weight_rays=True):
    max_rays_num = max([data.shape[1] for data in center_data])
    if max_rays_num == 0:
        return None
    total_rays = []
    for data in center_data:
        if weight_rays:
            # The normal vector of our final rays is just (0, 0, -1) in the
            # coordinate system we define.
            data = weight_rays_gaussian(data, [0, 0, -1])
        # only keep polarization, intensity, and phase
        # could also keep detector position and angle separately
        # if we're curious!
        power_vals_mask = [6, 7, 8]
        power_vals = data[power_vals_mask]
        power_vals = power_vals.T
        # pad rows now
        total_vals = np.zeros((max_rays_num, 3))
        total_vals[:power_vals.shape[0]] = power_vals
        total_rays.append(total_vals)

    return np.array(total_rays)


def get_interferogram_frequency(outrays, frequencies, debug=True):
    theta = outrays[:, :, 0]
    intensity = outrays[:, :, 1]
    distance = outrays[:, :, 2]
    ex1 = np.sqrt(intensity) * np.cos(theta)
    ey1 = np.sqrt(intensity) * np.sin(theta)

    total_power = np.zeros(outrays.shape[0])
    for freq in tqdm(frequencies, disable=(not debug)):
        
        wavelength = c / freq
        phase = np.exp(1j * (distance * 2 * np.pi / wavelength))
        ex = ex1 * phase
        ey = ey1 * phase

        power = np.square(np.abs((ex.sum(axis=1)))) + np.square(
            np.abs((ey.sum(axis=1))))
        total_power += power
    return total_power


def get_interferograms(out_data, freqs, n_linear_det, det_spacing, det_size,
                       add_diffraction_effects=False, weight_rays=True):
    total_out_segments, det_points = segment_rays(out_data, n_linear_det,
                                                  det_spacing, det_size)
    reorganized_segments = []
    for j in range(len(det_points)):
        reorganized_segments.append([])
    for i in range(len(total_out_segments)):
        for j in range(len(total_out_segments[i])):
            data = total_out_segments[i][j]
            # if data is empty, do nothing
            reorganized_segments[j].append(data)

    if (add_diffraction_effects):
        # don't want this for now. Debugging
        raise Exception("Diffracton effects not supported.")
        assert len(freqs) == 1
        reorganized_segments = add_diffraction_path_lengths(
            reorganized_segments, freqs[0], n_linear_det, det_spacing, det_size)

    interferograms = []
    for j, det_center in enumerate(det_points):
        # get the max number of rays
        data_matrix = data_to_matrix(reorganized_segments[
            j], weight_rays=weight_rays)
        if (data_matrix is None):
            interferogram = np.zeros(len(reorganized_segments[0]))
        else:
            interferogram = get_interferogram_frequency(
                data_matrix, freqs, debug=False)
            # interferogram = rt.get_interferogram(data_matrix, (c / 150.))
        interferograms.append(interferogram)
    return interferograms


def get_and_combine_interferograms(
        all_data, freqs, n_linear_det, det_spacing, det_size, debug=True,
        add_diffraction_effects=False, weight_rays=True):
    total_interferograms = []
    for data in tqdm(all_data, disable=(not debug)):
        interferograms = get_interferograms(
            data, freqs, n_linear_det, det_spacing, det_size,
            add_diffraction_effects=add_diffraction_effects,
            weight_rays=weight_rays)
        total_interferograms.append(interferograms)
    total_interferograms = np.array(total_interferograms)
    return total_interferograms


def add_outrays(total_outrays, start_displacement, n_mirror_positions,
                y_max, n_rays, semaphore, debug):
    outrays = trace_rays(start_displacement, n_mirror_positions, y_max,
                         n_rays, debug=debug)
    total_outrays.put([outrays, start_displacement])
    semaphore.release()


def get_outrays_threaded(
        x_displacements, y_displacements, n_mirror_positions, y_max,
        n_rays=225, debug=False):
    processes = []
    max_processes = 8
    semaphore = Semaphore(max_processes)
    manager = Manager()
    total_outrays = manager.Queue()

    # re-do and assume that each (x, y) is a single point.
    for (x, y) in zip(x_displacements, y_displacements):
        start_displacement = (x, y)
        semaphore.acquire()
        process = Process(target=add_outrays, args=(
            total_outrays, start_displacement, n_mirror_positions,
            y_max, n_rays, semaphore, debug),
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
    outrays_list = []
    start_displacement_list = []
    while (total_outrays.qsize() != 0):
        outrays, displacement = total_outrays.get()
        outrays_list.append(outrays)
        start_displacement_list.append(displacement)
    return outrays_list, start_displacement_list


def process_interferogram_discrete(
        outrays, frequency, total_interferograms, n_linear_det, det_spacing,
        det_size, semaphore):
    interferogram_sum_frequency = np.sum(get_and_combine_interferograms(
        outrays, np.arange(frequency, frequency + 1, 1), n_linear_det,
        det_spacing, det_size, debug=False, add_diffraction_effects=False),
                                         axis=0)
    total_interferograms.put([frequency, interferogram_sum_frequency])
    semaphore.release()


def get_interferogram_rays_at_z(ray_data, z):
    # Ray data should be of size (n_rays, n_mirror_positions)
    return [[get_rays_at_z(ray_data[i][j], z) for j in range(len(
        ray_data[i]))] for i in range(len(ray_data))]


def get_added_distance(dist_from_center, frequency, aperture_size):
    # added distance calculated to be (w^2 * 1.22 * lambda / d^2)
    wavelength = (c / frequency)  # in mm
    # we don't go above 300 Ghz usually.. can change if we do
    assert wavelength >= 1
    return ((dist_from_center ** 2) * 1.22 * wavelength) / (aperture_size ** 2)


def add_diffraction_path_lengths(reorganized_segments, frequency, n_linear_det,
                                 det_spacing, det_size):
    centers = get_centers(n_linear_det, det_spacing)
    assert len(reorganized_segments) == len(centers)
    diffraction_added_segments = []
    for center, segment in zip(centers, reorganized_segments):
        center = np.add(center, np.array(csims.FOCUS[:2]) * IN_TO_MM)
        diffraction_segment_rays = []
        for mirror_path_rays in segment:
            diffraction_rays = np.copy(mirror_path_rays)
            dist_from_aperture_center = np.sqrt(np.sum(np.square((
                diffraction_rays[[0, 1]] * IN_TO_MM).T - center), axis=1))
            assert np.all(dist_from_aperture_center <= (det_size / 2))
            diffraction_rays[8] += get_added_distance(
                dist_from_aperture_center, frequency, det_size)
            diffraction_segment_rays.append(diffraction_rays)
        diffraction_added_segments.append(diffraction_segment_rays)
    return diffraction_added_segments

def create_uniformly_spaced_points_circle(radius, num_points):
    # Determine the number of radial layers
    num_layers = int(np.sqrt(num_points) / 1.7) # Approximate number of layers

    # Initialize arrays for storing points
    x = []
    y = []

    # Generate points layer by layer
    for i in range(0, num_layers + 1):
        # Radial coordinate for this layer
        # r = np.sqrt(i / num_layers) * radius
        r = (i / num_layers) * radius

        # Number of points in this layer (proportional to radius)
        n_theta = int(2 * np.pi * r * num_layers / radius)  # Points proportional to circumference
        if n_theta == 0:  # Ensure at least one point per layer
            n_theta = 1

        # Angular coordinates
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

        # Convert to Cartesian coordinates and store
        x.extend(r * np.cos(theta))
        y.extend(r * np.sin(theta))

    # Convert to numpy arrays for easier handling
    x = np.array(x)
    y = np.array(y)
    return x, y


def postprocess_interferograms_discrete(ray_data, frequencies, n_linear_det,
                                        det_spacing, det_size, z_shift=0):

    processes = []
    max_processes = 8
    semaphore = Semaphore(max_processes)
    manager = Manager()
    total_interferograms = manager.Queue()
    ray_z_data = ray_data

    if z_shift != 0:
        print('Propagating rays to a z shift of %s inches.' % z_shift)
        ray_z_data = get_interferogram_rays_at_z(
            ray_data, csims.FOCUS[2] + z_shift)

    for freq in frequencies:
        semaphore.acquire()
        process = Process(target=process_interferogram_discrete, args=(
            ray_z_data, freq, total_interferograms, n_linear_det, det_spacing,
            det_size, semaphore))

        process.start()
        processes.append(process)

    # Wait for all the processes to finish before converting to a list.
    for i, process in enumerate(processes):
        process.join()

    # convert the shared queue to a list.
    total_interferogram_list = []
    frequency_list = []
    while (total_interferograms.qsize() != 0):
        frequency, interferograms = total_interferograms.get()
        total_interferogram_list.append(interferograms)
        frequency_list.append(frequency)

    # sort these myself.
    sort_inds = np.argsort(frequency_list)

    return np.array(total_interferogram_list)[sort_inds], \
        np.array(frequency_list)[sort_inds]
