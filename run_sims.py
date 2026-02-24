#!/usr/bin/env python3
import utils
from utils import get_outrays_threaded, create_uniformly_spaced_points_circle, postprocess_interferograms_discrete
import numpy as np
import pickle

def propagate_outrays_full_throw_high_res():
    FTS_stage_throw = 20.     # total throw extent in mm
    FTS_stage_step_size = .1  # FTS step size in mm
    n_mirror_positions = (2 * FTS_stage_throw / FTS_stage_step_size) + 1

    # same amount of points as before, but better spaced around a circle
    x_vals, y_vals = create_uniformly_spaced_points_circle(45, 100)

    total_outrays, displacements = get_outrays_threaded(
        x_vals, y_vals, n_mirror_positions, FTS_stage_throw,
        n_rays=225, debug=True)

    fname = "/Users/talford/data/FTS_sim_results/total_outrays_0_45_625_20260224.p"
    for outrays in total_outrays:
        with open(fname, 'ab') as output:
            pickle.dump(outrays, output, pickle.HIGHEST_PROTOCOL)

    pickle.dump(displacements, open(
        "/Users/talford/data/FTS_sim_results/displacements_0_45_625_20260224.p", "wb"))

    # save this for loading elsewhere
    print('finished!')

def main_20241121():
    propagate_outrays_full_throw_high_res()

def main_20260224():
    propagate_outrays_full_throw_high_res()

def postprocess_20241121():

    n_linear_det = 5
    det_spacing = 5
    det_size = 5

    data = []
    fname = "/data/talford/FTS_sim_results/total_outrays_0_45_625_20241120.p"

    with open(fname, 'rb') as f:
        try:
            while True:
                data.append(pickle.load(f))
        except EOFError:
            pass
    # It should take around 14h to run 21 of these data points. To cut down on
    # this I'll change the frequency resolution to 5 and run 11 data points.
    # so this should now take ~1.4 hours

    # In actuality it took 3ish hours
    freqs = np.arange(20, 305, 5)
    fname = "/data/talford/FTS_sim_results/all_discrete_interferograms_z" + (
        "0_45_625_25_20241120.p")
    z_coordinates = (1 / utils.IN_TO_MM) * np.linspace(-50, 75, 31)

    for z in z_coordinates:
        total_interferogram_list_at_z, frequency_list = \
            postprocess_interferograms_discrete(
                data, freqs, n_linear_det, det_spacing, det_size, z_shift=z)

        pickle.dump([z, frequency_list, total_interferogram_list_at_z], open(
            fname, "ab"), pickle.HIGHEST_PROTOCOL)

    # save this for loading elsewhere
    print('finished!')

def postprocess_20241204_mf():

    data = []
    fname = "/data/talford/FTS_sim_results/total_outrays_0_45_625_20241120.p"
    n_linear_det = 5
    det_spacing = 5
    det_size = 5.3

    with open(fname, 'rb') as f:
        try:
            while True:
                data.append(pickle.load(f))
        except EOFError:
            pass
    # It should take around 14h to run 21 of these data points. To cut down on
    # this I'll change the frequency resolution to 5 and run 11 data points.
    # so this should now take ~1.4 hours

    # In actuality it took 3ish hours
    freqs = np.arange(15, 305, 5)
    fname = "/data/talford/FTS_sim_results/all_discrete_interferograms_z" + (
        "0_45_625_25_20241204_lf.p")
    z_coordinates = (1 / utils.IN_TO_MM) * np.linspace(-50, 75, 31)

    for z in z_coordinates:
        total_interferogram_list_at_z, frequency_list = \
            postprocess_interferograms_discrete(
                data, freqs, n_linear_det, det_spacing, det_size, z_shift=z)

        pickle.dump([z, frequency_list, total_interferogram_list_at_z], open(
            fname, "ab"), pickle.HIGHEST_PROTOCOL)

    # save this for loading elsewhere
    print('finished!')


def postprocess_20241205_lf():

    data = []
    fname = "/data/talford/FTS_sim_results/total_outrays_0_45_625_20241120.p"
    n_linear_det = 5
    det_spacing = 5
    det_size = 14.0

    with open(fname, 'rb') as f:
        try:
            while True:
                data.append(pickle.load(f))
        except EOFError:
            pass
    # It should take around 14h to run 21 of these data points. To cut down on
    # this I'll change the frequency resolution to 5 and run 11 data points.
    # so this should now take ~1.4 hours

    # In actuality it took 3ish hours
    freqs = np.arange(15, 305, 5)
    fname = "/data/talford/FTS_sim_results/all_discrete_interferograms_z" + (
        "0_45_625_25_20241205_lf.p")
    z_coordinates = (1 / utils.IN_TO_MM) * np.linspace(-50, 75, 31)

    for z in z_coordinates:
        total_interferogram_list_at_z, frequency_list = \
            postprocess_interferograms_discrete(
                data, freqs, n_linear_det, det_spacing, det_size, z_shift=z)

        pickle.dump([z, frequency_list, total_interferogram_list_at_z], open(
            fname, "ab"), pickle.HIGHEST_PROTOCOL)

    # save this for loading elsewhere
    print('finished!')


def postprocess_20241210_uhf():

    data = []
    fname = "/data/talford/FTS_sim_results/total_outrays_0_45_625_20241120.p"
    n_linear_det = 5
    det_spacing = 5
    det_size = 4.75

    with open(fname, 'rb') as f:
        try:
            while True:
                data.append(pickle.load(f))
        except EOFError:
            pass
    # It should take around 14h to run 21 of these data points. To cut down on
    # this I'll change the frequency resolution to 5 and run 11 data points.
    # so this should now take ~1.4 hours

    # In actuality it took 3ish hours
    freqs = np.arange(15, 305, 5)
    fname = "/data/talford/FTS_sim_results/all_discrete_interferograms_z" + (
        "0_45_625_25_20241210_uhf.p")
    z_coordinates = (1 / utils.IN_TO_MM) * np.linspace(-50, 75, 31)

    for z in z_coordinates:
        total_interferogram_list_at_z, frequency_list = \
            postprocess_interferograms_discrete(
                data, freqs, n_linear_det, det_spacing, det_size, z_shift=z)

        pickle.dump([z, frequency_list, total_interferogram_list_at_z], open(
            fname, "ab"), pickle.HIGHEST_PROTOCOL)

    # save this for loading elsewhere
    print('finished!')


def postprocess_20260128():
    data = []
    fname = "/Users/talford/data/FTS_sim_results/total_outrays_0_45_625_20260128.p"
    n_linear_det = 9
    det_spacing = 5
    det_size = 5.3

    with open(fname, 'rb') as f:
        try:
            while True:
                data.append(pickle.load(f))
        except EOFError:
            pass

    freqs = np.arange(15, 305, 5)
    fname = "/Users/talford/data/FTS_sim_results/all_discrete_interferograms_20260128.p"
    # z_coordinates = (1 / utils.IN_TO_MM) * np.linspace(-100, 100, 25)
    # we already have 8 coordinates finished!
    z_coordinates = (1 / utils.IN_TO_MM) * np.linspace(-100, 100, 25)[9:]

    for z in z_coordinates:
        total_interferogram_list_at_z, frequency_list = \
            postprocess_interferograms_discrete(
                data, freqs, n_linear_det, det_spacing, det_size, z_shift=z)

        pickle.dump([z, frequency_list, total_interferogram_list_at_z], open(
            fname, "ab"), pickle.HIGHEST_PROTOCOL)

    # save this for loading elsewhere
    print('finished!')



if __name__ == '__main__':
    main_20260224()
    # postprocess_20241121()
    # postprocess_20241204_mf()
    # postprocess_20241205_lf()
    # postprocess_20241210_uhf()
