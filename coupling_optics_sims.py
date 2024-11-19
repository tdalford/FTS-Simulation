'''Code largely written by Grace Chesmore with a few modifications and bug
fixes by Tommy Alford '''

from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
# from scipy.spatial import distance
import fts_coupling_optics_geo as fts
from fts_coupling_optics_geo import *
import plotly.graph_objects as go
import plotly.io as pio
import random
from RayTraceFunctionsv2 import transformLG
pio.renderers.default = 'browser'

# shift_origin = [0., -266.21940725, -231.24377979]
shift_origin = [0, -276.84436351, -233.28894593]
# shift_origin = [0., 0., 0.]
tilt_angle = [np.pi - .19016, 0, 0]
# tilt_angle = [0, 0, 0]

FOCUS = [0, ((210+82) * mm_to_in + 1.848), -20.9]
CENTER_11 = [0, -.426, 0]


def snell_vec(n1, n2, N_surf, s1):
    # s1 is the incoming vector, pointing from the light source to the surface
    # N_surf is the normal of the surface

    s2 = (n1/n2) * np.cross(N_surf, (np.cross(-N_surf, s1))) \
        - N_surf*np.sqrt(1-(n1/n2)**2 * np.dot((np.cross(N_surf, s1)),
                                               (np.cross(N_surf, s1))))
    return s2


x = np.linspace(-(3), (3), 100)  # [in]
y = np.linspace(-(3), (3), 100)  # [in]
X, Y = np.meshgrid(x, y)

# Define all surfaces of lenses
Z_l1s1 = l1s1(X, Y)
Z_l1s2 = l1s2(X, Y)

Z_l2s1 = l2s1(X, Y)
Z_l2s2 = l2s2(X, Y)

Z_l3s1 = l3s1(X, Y)
Z_l3s2 = l3s2(X, Y)

Z_flat = flat(X, Y)

# These functions rotate a given surface and shift it to its appropriate location
# in the FTS coupling optics setup. I gave an  option to shift, or not shift, because
# if you want to switch references frames of a normal vector, you do NOT want to shift it,
# you just want to rotate it.  Here I shift it because i want to visualize it in the
# complete setup:
shift = 1
Xt11, Yt11, Zt11 = ref_11_to_tele(X, Y, Z_l1s1, shift)
Xt12, Yt12, Zt12 = ref_12_to_tele(X, Y, Z_l1s2, shift)

Xt21, Yt21, Zt21 = ref_21_to_tele(X, Y, Z_l2s1, shift)
Xt22, Yt22, Zt22 = ref_22_to_tele(X, Y, Z_l2s2, shift)

Xtflat, Ytflat, Ztflat = ref_flat_to_tele(X, Y, Z_flat, shift)
Xt31, Yt31, Zt31 = ref_31_to_tele(X, Y, Z_l3s1, shift)
Xt32, Yt32, Zt32 = ref_32_to_tele(X, Y, Z_l3s2, shift)


def plot_surface(x1, y1, z1, x2, y2, z2, fig):
    x1_tot, y1_tot, z1_tot = ([], [], [])
    x2_tot, y2_tot, z2_tot = ([], [], [])

    for l in range(0, 100, 1):
        x1_tot.extend(x1[:, l])
        y1_tot.extend(y1[:, l])
        z1_tot.extend(z1[:, l])

        x2_tot.extend(x2[:, l])
        y2_tot.extend(y2[:, l])
        z2_tot.extend(z2[:, l])

    # l = int(len(y1) / 2)
    line1 = go.Mesh3d(x=x1_tot, y=y1_tot, z=z1_tot, color='blue', opacity=.4)
    line2 = go.Mesh3d(x=x2_tot, y=y2_tot, z=z2_tot, color='blue', opacity=.4)
    fig.add_trace(line1)
    fig.add_trace(line2)


def plot_surface(x, y, z, fig, start, stop, color, half_index=50,
                 shift=[0, 0, 0], tilt=[0, 0, 0], fac=1, alpha=.1):

    x_tot1, y_tot1, z_tot1, x_tot2, y_tot2, z_tot2 = ([], [], [], [], [], [])

    for l in range(start, stop, 1):
        x_new_1, y_new_1, z_new_1 = transformLG(
            fac * x[:half_index, l], fac * y[:half_index, l],
            fac * z[:half_index, l] * -1, shift, tilt)

        x_new_2, y_new_2, z_new_2 = transformLG(
            fac * x[half_index:, l], fac * y[half_index:, l],
            fac * z[half_index:, l] * -1, shift, tilt)

        # x_tot.extend(x[:half_index, l])
        # y_tot.extend(y[:half_index, l])
        # z_tot1.extend(z[:half_index, l])

        # x_tot2.extend(x[half_index:, l])
        # y_tot2.extend(y[half_index:, l])
        # z_tot2.extend(z[half_index:, l])

        x_tot1.extend(x_new_1)
        y_tot1.extend(y_new_1)
        z_tot1.extend(z_new_1)

        x_tot2.extend(x_new_2)
        y_tot2.extend(y_new_2)
        z_tot2.extend(z_new_2)

    # Change x and z directions to match FTS defined
    shape = go.Mesh3d(x=z_tot1, y=y_tot1, z=x_tot1, color=color,
                      opacity=alpha, alphahull=-1, delaunayaxis='y')
    shape2 = go.Mesh3d(x=z_tot2, y=y_tot2, z=x_tot2, color=color,
                       opacity=alpha, alphahull=-1, delaunayaxis='y')
    fig.add_trace(shape)
    fig.add_trace(shape2)


def plot_ray(x_vals, y_vals, z_vals, fig, color, alpha, shift=[0, 0, 0],
             tilt=[0, 0, 0], fac=1):

    x_new, y_new, z_new = transformLG(fac * np.array(x_vals),
                                      fac * np.array(y_vals),
                                      fac * np.array(z_vals) * -1, shift, tilt)
    line = go.Scatter3d(x=z_new, y=y_new, z=x_new, mode='lines',
                        showlegend=False, line=dict(color=color),
                        opacity=alpha)
    fig.add_trace(line)


def norm(v):
    return np.sqrt(np.sum(np.square(v)))


def normalize(v):
    return v / norm(v)


def convert_to_ray_mcmahon(out_arr):
    out_point = out_arr[[0, 1, 2]]
    out_vector = out_arr[[8, 9, 10]]
    distance = out_arr[3]

    # we need to convert from mm to inches here!
    factor = (1 / mm_to_in)

    # switch the x and z coordinate of these!
    out_point = factor * np.flip(out_point * [1, 0, 1])
    out_vector = np.flip(
        normalize(factor * out_vector * [1, -1, 1]))

    # need values for polarization, intensity
    # polarization_angle = random.random() * 2 * np.pi
    polarization_angle = .123
    intensity = 1.0

    ray = [polarization_angle, intensity, out_point, out_vector,
           distance * factor]
    return ray


# add a loss tangent to this index of refraction
n_lens = 1.517  # + 1e-6 * np.complex(0, 1)  # index of refraction for HDPE
n_vac = 1       # index of refraction vacuum


def gaussian(x, mu, sigma):
    return np.exp((-(x - mu) ** 2) / (2 * (sigma ** 2)))


def inverse_gaussian_function(x, mu, sigma):
    # assumes that x <= mu, no normalization
    return mu - np.sqrt(-1 * (sigma ** 2) * 2 * np.log(x))


def run_rays_through_coupling_optics_reversed(
        P_rx, tele_geo, col, fig, starting_rays=None, n_linear=40,
        alpha=.04, theta_bound=.1, plot=True, y_ap=CENTER_11[1]):

    # y_ap = -2.34 # this is where you want your rays to end (in y). You can definitely change this! I was just guessing.
    # y_ap = -20.9
    N_linear = tele_geo.N_scan  # number of rays in 1D scan
    N_linear = n_linear
    # alph = 0.04  # transparency of plotted lines
    alph = alpha

    # Step 1:  grid the plane of rays shooting out of FTS
    # In your code, you would just input the direction of a given ray
    # and probably skip this step. I'm just creating a very tidy array
    # of rays but you will already have that leaving the FTS.
    # theta_bound = .1

    # z_range = np.linspace(np.cos(theta_bound), 1, N_linear)

    # set sigma of the beam manually for now.
    # sigma = .01
    # gaussian_z_min = gaussian(np.cos(theta_bound), 1, sigma)
    # gaussian_z_max = gaussian(1, 1, sigma)  # should just be 1
    # assert gaussian_z_max == 1
    # # skip the first linearly spaced value
    # gaussian_z_vals = np.linspace(gaussian_z_min, gaussian_z_max, n_linear)
    # z_range = inverse_gaussian_function(gaussian_z_vals, 1, sigma)
    # phi_range = np.linspace(0, 2 * np.pi, N_linear)

    # z_vals, phi_vals = np.meshgrid(z_range, phi_range)
    # z_vals = np.ravel(z_vals)
    # phi_vals = np.ravel(phi_vals)

    # # Step 2: calculate the position + local surface normal for the dish
    # n_pts = len(phi_vals)
    # # out = np.zeros((17, n_pts))
    # out = [[], [], [], [], [], [], [], [], [], [], []]

    # for ii in range(n_pts):
    #     z_val = z_vals[ii]
    #     phi_val = phi_vals[ii]

    #     # Direction of ray away from the starting point
    #     sinthet = np.sqrt(1 - z_val ** 2)
    #     r_hat = [sinthet * np.cos(phi_val), sinthet * np.sin(phi_val), z_val]

    if (starting_rays is None):
        theta = np.linspace((0) - theta_bound, (0) + theta_bound, N_linear)
        phi = np.linspace((0) - (np.pi/2), (0) + (np.pi/2), N_linear)

        theta, phi = np.meshgrid(theta, phi)
        theta = np.ravel(theta)
        phi = np.ravel(phi)

        # Step 2: calculate the position + local surface normal for the dish
        n_pts = len(theta)
    else:
        n_pts = len(starting_rays)

    out = [[], [], [], [], [], [], [], [], [], [], []]

    for ii in range(n_pts):

        if (starting_rays is None):
            th = theta[ii]
            ph = phi[ii]

            # Direction of ray away from the starting point
            r_hat = [np.sin(th) * np.cos(ph), np.sin(th)
                     * np.sin(ph), np.cos(th)]

            alpha = r_hat[0]
            beta = r_hat[1]
            gamma = r_hat[2]

            # I refer to the starting point of the rays as 'rx' throughout the
            # code. So when you see that, that's what that means.
            x_rx = P_rx[0]
            y_rx = P_rx[1]
            z_rx = P_rx[2]
        else:
            x_rx, y_rx, z_rx = starting_rays[ii][2]
            P_rx = [x_rx, y_rx, z_rx]
            alpha, beta, gamma = starting_rays[ii][3]

        # To hit a surface, I use a root finder. This tells the ray to search
        # for a distance 't' steps away until it hits a defined surface. The
        # variable 't' is basically how many normal vectors do I need to travel
        # before I hit the desired surface. I do this for every surface
        # throughout this function.

        def root_32(t):
            x = x_rx + alpha * t
            y = y_rx + beta * t
            z = z_rx + gamma * t
            x32, y32, z32 = ref_tele_to_32(
                x, y, z, 1
            )
            z_32 = l3s2(x32, y32)
            if np.isnan(z_32) == True:
                z_32 = 0
            root = z32 - z_32
            return root

        try:
            t_32 = optimize.brentq(root_32, 0.01, 500)  # [in.]
        except ValueError:
            # print(ii)
            continue
        # f(a) and f(b) need to be opposite signs

        # Location of where ray hits M1
        x_32 = P_rx[0] + alpha * t_32
        y_32 = P_rx[1] + beta * t_32
        z_32 = P_rx[2] + gamma * t_32
        P_32 = np.array([x_32, y_32, z_32])

        ###### in lens surface reference frame ##########################
        x_32_temp, y_32_temp, z_32_temp = ref_tele_to_32(
            x_32, y_32, z_32, 1
        )  # P_32 temp
        x_rx_temp, y_rx_temp, z_rx_temp = ref_tele_to_32(
            x_rx, y_rx, z_rx, 1
        )  # P_rx temp
        norm = d_z(x_32_temp, y_32_temp, lens3_surf2, "aspheric")
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))

        vec_rx_32 = np.array([x_32_temp, y_32_temp, z_32_temp]) - np.array(
            [x_rx_temp, y_rx_temp, z_rx_temp]
        )
        dist_rx_32 = np.sqrt(np.sum(vec_rx_32 ** 2))
        tan_rx_32 = vec_rx_32 / dist_rx_32

        tan_og_lens = snell_vec(n_vac, n_lens, N_hat, tan_rx_32)

        ###### Transform back into telescope reference frame ##########################
        N_hat_t_x, N_hat_t_y, N_hat_t_z = ref_32_to_tele(
            N_hat[0], N_hat[1], N_hat[2], 0
        )
        N_hat_t = [N_hat_t_x, N_hat_t_y, N_hat_t_z]
        tan_rx_32_t_x, tan_rx_32_t_y, tan_rx_32_t_z = ref_32_to_tele(
            tan_rx_32[0], tan_rx_32[1], tan_rx_32[2], 0
        )
        tan_rx_32_t = [tan_rx_32_t_x, tan_rx_32_t_y, tan_rx_32_t_z]
        tan_og_t_x, tan_og_t_y, tan_og_t_z = ref_32_to_tele(
            tan_og_lens[0], tan_og_lens[1], tan_og_lens[2], 0
        )
        tan_og_t = [tan_og_t_x, tan_og_t_y, tan_og_t_z]  # outgoing ray vector

        # I've commented these out, but if you want to view the outgoing ray direction,
        # normal vector of surface, incoming ray direction, and  of this surface
        # you can uncomment and plot the normal vector of the 13th ray.

        # if ii==13:
        #     plt.plot([y_32,y_32+(5*tan_og_t[1])], [z_32,z_32+(5*tan_og_t[2])], "-",color = 'r')
        #     plt.plot([y_32,y_32+(5*N_hat_t[1])], [z_32,z_32+(5*N_hat_t[2])], "-",color = 'k')
        #     plt.plot([y_32,y_32-(5*tan_rx_32_t[1])], [z_32,z_32-(5*tan_rx_32_t[2])], "-",color = 'g')

        ################
        ################
        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        def root_31(t):
            x = x_32 + alpha * t
            y = y_32 + beta * t
            z = z_32 + gamma * t
            x31, y31, z31 = ref_tele_to_31(
                x, y, z, 1
            )
            z_31 = l3s1(x31, y31)
            if np.isnan(z_31) == True:
                z_31 = 0
            root = z31 - z_31
            return root
        try:
            t_31 = optimize.brentq(root_31, 0.01, 500)
        except ValueError:
            # print(ii)
            continue
        # Location of where ray hits M1
        x_31 = P_32[0] + alpha * t_31
        y_31 = P_32[1] + beta * t_31
        z_31 = P_32[2] + gamma * t_31
        P_31 = np.array([x_31, y_31, z_31])

        ###### in lens surface reference frame ##########################
        x_31_temp, y_31_temp, z_31_temp = ref_tele_to_31(
            x_31, y_31, z_31, 1
        )  # P_m2 temp
        x_32_temp, y_32_temp, z_32_temp = ref_tele_to_31(
            x_32, y_32, z_32, 1
        )  # P_rx temp
        norm = d_z(x_31_temp, y_31_temp, lens3_surf1, "aspheric")
        norm_temp = np.array([-norm[0], -norm[1], 1])

        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))
        vec_32_31 = np.array([x_31_temp, y_31_temp, z_31_temp]) - np.array(
            [x_32_temp, y_32_temp, z_32_temp]
        )
        dist_32_31 = np.sqrt(np.sum(vec_32_31 ** 2))
        tan_32_31 = vec_32_31 / dist_32_31

        tan_og_lens = snell_vec(n_lens, n_vac, -N_hat, tan_32_31)

        ###### Transform back into telescope reference frame ##########################
        N_hat_t_x, N_hat_t_y, N_hat_t_z = ref_31_to_tele(
            N_hat[0], N_hat[1], N_hat[2], 0
        )
        N_hat_t = [N_hat_t_x, N_hat_t_y, N_hat_t_z]
        tan_32_31_t_x, tan_32_31_t_y, tan_32_31_t_z = ref_31_to_tele(
            tan_32_31[0], tan_32_31[1], tan_32_31[2], 0
        )
        tan_32_31_t = [tan_32_31_t_x, tan_32_31_t_y, tan_32_31_t_z]
        tan_og_t_x, tan_og_t_y, tan_og_t_z = ref_31_to_tele(
            tan_og_lens[0], tan_og_lens[1], tan_og_lens[2], 0
        )
        tan_og_t = [tan_og_t_x, tan_og_t_y, tan_og_t_z]

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]
        ##################################################

        def root_flat(t):

            x = x_31 + alpha * t
            y = y_31 + beta * t
            z = z_31 + gamma * t
            xflat, yflat, zflat = ref_tele_to_flat(
                x, y, z, 1
            )
            z_flat = flat(xflat, yflat)
            if np.isnan(z_flat) == True:
                z_flat = 0
            root = zflat - z_flat
            return root

        try:
            t_flat = optimize.brentq(root_flat, 0.03, 50)
        except ValueError:
            # print(ii)
            continue

        # Location of where ray hits M2
        x_flat = x_31 + alpha * t_flat
        y_flat = y_31 + beta * t_flat
        z_flat = z_31 + gamma * t_flat
        P_flat = np.array([x_flat, y_flat, z_flat])

        ###### in lens surface reference frame ##########################
        x_flat_temp, y_flat_temp, z_flat_temp = ref_tele_to_flat(
            x_flat, y_flat, z_flat, 1
        )  # P_m2 temp
        x_31_temp, y_31_temp, z_31_temp = ref_tele_to_flat(
            x_31, y_31, z_31, 1
        )  # P_rx temp
        norm_temp = np.array([0, 0, 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))

        vec_31_flat = np.array([x_flat_temp, y_flat_temp, z_flat_temp]) - np.array(
            [x_31_temp, y_31_temp, z_31_temp]
        )
        dist_31_flat = np.sqrt(np.sum(vec_31_flat ** 2))
        tan_31_flat = vec_31_flat / dist_31_flat

        tan_og_lens = 2 * (np.dot(N_hat, - tan_31_flat)
                           * (N_hat)) - (-tan_31_flat)

        ###### Transform back into telescope reference frame ##########################
        N_hat_t_x, N_hat_t_y, N_hat_t_z = ref_flat_to_tele(
            N_hat[0], N_hat[1], N_hat[2], 0
        )
        N_hat_t = [N_hat_t_x, N_hat_t_y, N_hat_t_z]
        tan_31_flat_t_x, tan_31_flat_t_y, tan_31_flat_t_z = ref_flat_to_tele(
            tan_31_flat[0], tan_31_flat[1], tan_31_flat[2], 0
        )
        tan_31_flat_t = [tan_31_flat_t_x, tan_31_flat_t_y, tan_31_flat_t_z]
        tan_og_t_x, tan_og_t_y, tan_og_t_z = ref_flat_to_tele(
            tan_og_lens[0], tan_og_lens[1], tan_og_lens[2], 0
        )
        tan_og_t = [tan_og_t_x, tan_og_t_y, tan_og_t_z]

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        # print(
        #     f'intersect at lens flat from reversed = {np.around(P_flat, decimals=5)}')

        def root_22(t):
            x = P_flat[0] + alpha * t
            y = P_flat[1] + beta * t
            z = P_flat[2] + gamma * t

            x22, y22, z22 = ref_tele_to_22(
                x, y, z, 1
            )
            z_22 = l2s2(x22, y22)
            if np.isnan(z_22) == True:
                z_22 = 0
            root = z22 - z_22
            return root

        try:
            t_22 = optimize.brentq(root_22, 0.05, 10)
        except ValueError:
            # print(ii)
            continue

        # Location of where ray hits lens surface
        x_22 = P_flat[0] + alpha * t_22
        y_22 = P_flat[1] + beta * t_22
        z_22 = P_flat[2] + gamma * t_22
        P_22 = np.array([x_22, y_22, z_22])

        ###### in lens surface reference frame ##########################
        x_22_temp, y_22_temp, z_22_temp = ref_tele_to_22(
            x_22, y_22, z_22, 1
        )  # P_m2 temp
        x_flat_temp, y_flat_temp, z_flat_temp = ref_tele_to_22(
            x_flat, y_flat, z_flat, 1
        )  # P_rx temp
        norm = d_z(x_22_temp, y_22_temp, lens2_surf2, "spheric")
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))

        vec_flat_22 = np.array([x_22_temp, y_22_temp, z_22_temp]) - np.array(
            [x_flat_temp, y_flat_temp, z_flat_temp]
        )
        dist_flat_22 = np.sqrt(np.sum(vec_flat_22 ** 2))
        tan_flat_22 = vec_flat_22 / dist_flat_22

        # Use Snell's Law to find angle of outgoing ray:
        tan_og_lens = snell_vec(n_vac, n_lens, N_hat, tan_flat_22)

        ###### Transform back into telescope reference frame ##########################
        N_hat_t_x, N_hat_t_y, N_hat_t_z = ref_22_to_tele(
            N_hat[0], N_hat[1], N_hat[2], 0
        )
        N_hat_t = [N_hat_t_x, N_hat_t_y, N_hat_t_z]
        tan_flat_22_t_x, tan_flat_22_t_y, tan_flat_22_t_z = ref_22_to_tele(
            tan_flat_22[0], tan_flat_22[1], tan_flat_22[2], 0
        )
        tan_flat_22_t = [tan_flat_22_t_x, tan_flat_22_t_y, tan_flat_22_t_z]
        tan_og_t_x, tan_og_t_y, tan_og_t_z = ref_22_to_tele(
            tan_og_lens[0], tan_og_lens[1], tan_og_lens[2], 0
        )
        tan_og_t = [tan_og_t_x, tan_og_t_y, tan_og_t_z]

        # print(
        #     f'intersect at lens 22 from reversed = {np.around(P_22, decimals=5)}')
        # print(
        #     f'vector going from flat to 22 reversed = {np.around(tan_flat_22, decimals=5)}')
        # print(
        #     f'vector from 22 to 21 reversed in global frame = {np.around(tan_og_t, decimals=5)}')

        ##################################################
        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        def root_21(t):
            x = P_22[0] + alpha * t
            y = P_22[1] + beta * t
            z = P_22[2] + gamma * t

            x21, y21, z21 = ref_tele_to_21(
                x, y, z, 1
            )  # take ray end coordinates and convert to M1 coordinates
            z_21 = l2s1(x21, y21)  # Z of mirror 1 in M1 coordinates
            if np.isnan(z_21) == True:
                z_21 = 0
            root = z21 - z_21
            return root

        try:
            t_21 = optimize.brentq(root_21, .01, 30)
        except ValueError:
            # print(ii)
            continue

        # Location of where ray hits M1
        x_21 = P_22[0] + alpha * t_21
        y_21 = P_22[1] + beta * t_21
        z_21 = P_22[2] + gamma * t_21
        P_21 = np.array([x_21, y_21, z_21])

        # print(80 * '-')
        # print(f'P22 intersect = {np.around(P_22, decimals=5)}')
        # print(f'initial normal vec = {np.around(tan_og_t, decimals=5)}')
        # print(f'P21 intersect = {np.around(P_21, decimals=5)}')

        ###### in lens surface reference frame ##########################
        x_21_temp, y_21_temp, z_21_temp = ref_tele_to_21(
            x_21, y_21, z_21, 1
        )  # P_m2 temp
        # print(
        #     f'vec at 21 in 21 surface frame = {np.around([x_21_temp, y_21_temp, z_21_temp], decimals=5)}')
        x_22_temp, y_22_temp, z_22_temp = ref_tele_to_21(
            x_22, y_22, z_22, 1
        )  # P_rx temp

        # print(
        #     f'vec at 22 in 21 surface frame = {np.around([x_22_temp, y_22_temp, z_22_temp], decimals=5)}')
        norm = d_z(x_21_temp, y_21_temp, lens2_surf1, "spheric")
        # print(f'norm at 21 = {np.around(norm, 5)}')
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))

        vec_22_21 = np.array([x_21_temp, y_21_temp, z_21_temp]) - np.array(
            [x_22_temp, y_22_temp, z_22_temp]
        )
        dist_22_21 = np.sqrt(np.sum(vec_22_21 ** 2))
        tan_22_21 = vec_22_21 / dist_22_21

        # Use Snell's Law to find angle of outgoing ray:

        # print(
        #     f'snells law at 21 to go to 12: n_lens = {n_lens}. \n N_hat = {np.around(N_hat, 5)}. tan_22_21 = {np.around(tan_22_21, 5)}')
        # I think this is the one I need to change!
        # changing from n_vac, n_lens to n_lens, n_vac
        tan_og_lens = snell_vec(n_lens, n_vac, N_hat, tan_22_21)

        ###### Transform back into telescope reference frame ##################
        N_hat_t_x, N_hat_t_y, N_hat_t_z = ref_21_to_tele(
            N_hat[0], N_hat[1], N_hat[2], 0
        )
        N_hat_t = [N_hat_t_x, N_hat_t_y, N_hat_t_z]
        tan_22_21_t_x, tan_22_21_t_y, tan_22_21_t_z = ref_21_to_tele(
            tan_22_21[0], tan_22_21[1], tan_22_21[2], 0
        )
        tan_22_21_t = [tan_22_21_t_x, tan_22_21_t_y, tan_22_21_t_z]
        tan_og_t_x, tan_og_t_y, tan_og_t_z = ref_21_to_tele(
            tan_og_lens[0], tan_og_lens[1], tan_og_lens[2], 0
        )
        tan_og_t = [tan_og_t_x, tan_og_t_y, tan_og_t_z]

        # print(80 * '-')

        # print(
        #     f'vector going from 22 to 21 reversed = {np.around(tan_22_21, decimals=5)}')

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        def root_12(t):

            x = x_21 + alpha * t
            y = y_21 + beta * t
            z = z_21 + gamma * t

            x12, y12, z12 = ref_tele_to_12(
                x, y, z, 1
            )  # Convert ray's endpoint into M2 coordinates

            z_12 = l1s2(x12, y12)  # Z of mirror in M2 coordinates
            if np.isnan(z_12) == True:
                z_12 = 0
            root = z12 - z_12
            return root

        try:
            t_12 = optimize.brentq(root_12, 1e-3, 90)
        except ValueError:
            # print(ii)
            continue

        x_12 = P_21[0] + alpha * t_12
        y_12 = P_21[1] + beta * t_12
        z_12 = P_21[2] + gamma * t_12
        P_12 = np.array([x_12, y_12, z_12])

        ###### in lens surface reference frame ##########################
        x_12_temp, y_12_temp, z_12_temp = ref_tele_to_12(
            x_12, y_12, z_12, 1
        )  # P_m2 temp
        x_21_temp, y_21_temp, z_21_temp = ref_tele_to_12(
            x_21, y_21, z_21, 1
        )  # P_rx temp
        norm = d_z(x_12_temp, y_12_temp, lens1_surf2, "aspheric")
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))

        vec_21_12 = np.array([x_12_temp, y_12_temp, z_12_temp]) - np.array(
            [x_21_temp, y_21_temp, z_21_temp]
        )
        dist_21_12 = np.sqrt(np.sum(vec_21_12 ** 2))
        tan_21_12 = vec_21_12 / dist_21_12

        tan_og_lens = snell_vec(n_vac, n_lens, N_hat, tan_21_12)

        ###### Transform back into telescope reference frame ##########################
        N_hat_t_x, N_hat_t_y, N_hat_t_z = ref_12_to_tele(
            N_hat[0], N_hat[1], N_hat[2], 0
        )
        N_hat_t = [N_hat_t_x, N_hat_t_y, N_hat_t_z]
        tan_21_12_t_x, tan_21_12_t_y, tan_21_12_t_z = ref_12_to_tele(
            tan_21_12[0], tan_21_12[1], tan_21_12[2], 0
        )
        tan_21_12_t = [tan_21_12_t_x, tan_21_12_t_y, tan_21_12_t_z]
        tan_og_t_x, tan_og_t_y, tan_og_t_z = ref_12_to_tele(
            tan_og_lens[0], tan_og_lens[1], tan_og_lens[2], 0
        )
        tan_og_t = [tan_og_t_x, tan_og_t_y, tan_og_t_z]

        # print(
        #     f'intersect at lens 12 from reversed = {np.around(P_12, decimals=5)}')
        # print(
        #     f'vector going from 21 to 12 reversed = {np.around(tan_21_12, decimals=5)}')

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        def root_11(t):

            x = x_12 + alpha * t  # the variable 't' I explained above ^
            y = y_12 + beta * t
            z = z_12 + gamma * t

            x11, y11, z11 = ref_tele_to_11(
                x, y, z, 1
            )  # Convert ray's endpoint into the surface's reference frame

            z_11 = l1s1(x11, y11)  # Z of surface in surface's reference frame
            # I just put this in here to make sure nothing crazy happens (i.e. avoid nan values)
            if np.isnan(z_11) == True:
                z_11 = 0
            root = z11 - z_11
            return root

        try:
            t_11 = optimize.brentq(root_11, 1e-3, 90)
        except ValueError:
            # print(ii)
            continue

        # Location of where ray hits lens 1 surface 1
        x_11 = x_12 + alpha * t_11
        y_11 = y_12 + beta * t_11
        z_11 = z_12 + gamma * t_11
        P_11 = np.array([x_11, y_11, z_11])

        ###### Transform into lens 1 surface 1 coordinates ##########################
        x_11_temp, y_11_temp, z_11_temp = ref_tele_to_11(
            x_11, y_11, z_11, 1
        )
        x_12_temp, y_12_temp, z_12_temp = ref_tele_to_11(
            x_12, y_12, z_12, 1)  # P_rx temp
        norm = d_z(x_11_temp, y_11_temp, lens1_surf1, "spheric")
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))

        vec_12_11 = np.array([x_11_temp, y_11_temp, z_11_temp]) - np.array(
            [x_12_temp, y_12_temp, z_12_temp]
        )
        dist_12_11 = np.sqrt(np.sum(vec_12_11 ** 2))
        tan_12_11 = vec_12_11 / dist_12_11

        # Use Snell's Law to find angle of outgoing ray:
        tan_og_lens = snell_vec(n_lens, n_vac, N_hat, tan_12_11)

        ###### Transform back into telescope reference frame ##########################
        N_hat_t_x, N_hat_t_y, N_hat_t_z = ref_11_to_tele(
            N_hat[0], N_hat[1], N_hat[2], 0
        )
        N_hat_t = [-N_hat_t_x, -N_hat_t_y, -N_hat_t_z]
        tan_12_11_t_x, tan_12_11_t_y, tan_12_11_t_z = ref_11_to_tele(
            tan_12_11[0], tan_12_11[1], tan_12_11[2], 0
        )
        tan_12_11_t = [tan_12_11_t_x, tan_12_11_t_y, tan_12_11_t_z]
        tan_og_t_x, tan_og_t_y, tan_og_t_z = ref_11_to_tele(
            tan_og_lens[0], tan_og_lens[1], tan_og_lens[2], 0
        )
        tan_og_t = [tan_og_t_x, tan_og_t_y, tan_og_t_z]

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        dist_11_ap = abs((y_ap - P_11[1]) / tan_og_t[1])

        total_path_length = (
            t_11*n_lens + t_12 + t_21*n_lens + t_22 +
            t_flat + t_31*n_lens + t_32 + dist_11_ap
        )
        dist_in_lenses = t_11 + t_21 + t_31

        # print(t_11, t_12, t_21, t_22, t_flat, t_31,
        #       t_32, dist_11_ap)

        pos_ap = P_11 + dist_11_ap * np.array(tan_og_t)

        # Estimate theta, phi of outgoing initial ray
        de_ve = np.arctan(tan_rx_32_t[2] / (-tan_rx_32_t[1]))
        de_ho = np.arctan(
            tan_rx_32_t[0] / np.sqrt(tan_rx_32_t[2] ** 2 + tan_rx_32_t[1] ** 2)
        )

        #################################################
        ## Plotting lenses and rays #####################
        x_points = [x_rx, x_32, x_31, x_flat,
                    x_22, x_21, x_12, x_11, pos_ap[0]]
        y_points = [y_rx, y_32, y_31, y_flat,
                    y_22, y_21, y_12, y_11, pos_ap[1]]
        z_points = [z_rx, z_32, z_31, z_flat,
                    z_22, z_21, z_12, z_11, pos_ap[2]]

        # These rays aren't going to hit the FTS and are really incorrect here
        # (surfaces defined don't really have boundaries so weird things can happen), so we discard them
        if np.abs(pos_ap[0]) > 2 or np.abs(pos_ap[2]) > 2:
            continue
        # print('plotting...')
        if (plot):
            for i in range(len(x_points) - 1):
                plot_ray([x_points[i], x_points[i + 1]], [y_points[i], y_points[i + 1]],
                         [z_points[i], z_points[i + 1]], fig, col, alph,
                         shift=shift_origin, tilt=tilt_angle,
                         fac=(1 / mm_to_in))

        # Write out
        # print('writing out value %s' %ii)
        out[0].append(pos_ap[0])
        out[1].append(pos_ap[1])
        out[2].append(pos_ap[2])

        out[3].append(total_path_length)
        power_loss = np.exp(-1 * dist_in_lenses * (1 / mm_to_in) * 3e-4 * 2 * (
            np.pi * n_lens))
        out[4].append(power_loss)
#         out[4, ii] = np.exp(
#             (-0.5)
#             * ((de_ve) ** 2 + (de_ho) ** 2)
#             / ((24 * np.pi / 180) / (np.sqrt(8 * np.log(2)))) ** 2 #amplitude
       # ) # Setting FWHP = 24 degrees, you should change this

        out[5].append(N_hat_t[0])
        out[6].append(N_hat_t[1])
        out[7].append(N_hat_t[2])

        out[8].append(tan_og_t[0])
        out[9].append(tan_og_t[1])
        out[10].append(tan_og_t[2])

    if (plot):
        plot_surface(Xt11, Yt11, Zt11, fig, 0, 100, 'teal', half_index=50,
                     shift=shift_origin, tilt=tilt_angle, fac=(1 / mm_to_in))
        plot_surface(Xt12, Yt12, Zt12, fig, 0, 100, 'teal',
                     shift=shift_origin, tilt=tilt_angle, fac=(1 / mm_to_in))

        plot_surface(Xt21, Yt21, Zt21, fig, 0, 100, 'teal',
                     shift=shift_origin, tilt=tilt_angle, fac=(1 / mm_to_in))
        plot_surface(Xt22, Yt22, Zt22, fig, 0, 100, 'teal',
                     shift=shift_origin, tilt=tilt_angle, fac=(1 / mm_to_in))

        plot_surface(Xtflat, Ytflat, Ztflat, fig, 0, 100, 'silver',
                     half_index=100, shift=shift_origin, tilt=tilt_angle,
                     fac=(1 / mm_to_in), alpha=.3)

        plot_surface(Xt31, Yt31, Zt31, fig, 0, 100, 'teal', half_index=100,
                     shift=shift_origin, tilt=tilt_angle, fac=(1 / mm_to_in))
        plot_surface(Xt32, Yt32, Zt32, fig, 0, 100, 'teal', half_index=100,
                     shift=shift_origin, tilt=tilt_angle, fac=(1 / mm_to_in))

    return np.array(out)  # , [x_points, y_points, z_points]


def get_final_rays_reversed(shift, n_linear, theta_bound=.3,
                            y_ap=CENTER_11[1]):
    assert y_ap <= CENTER_11[1]
    # we want the shift in mm so now let's convert to inches
    # shift = np.multiply(shift, mm_to_in)
    # print(shift)
    start_position = FOCUS
    new_start = np.add(start_position, shift)

    out = run_rays_through_coupling_optics_reversed(
        new_start, fts_geo, None, None, n_linear=n_linear,
        theta_bound=theta_bound, plot=False, y_ap=y_ap)

    start_rays_mcmahon = [convert_to_ray_mcmahon(out_arr) for out_arr in out.T]
    return start_rays_mcmahon


def run_rays_through_coupling_optics_forwards(
        P_rx, tele_geo, col, fig, starting_rays=None, n_linear=40, alpha=.04,
        theta_bound=.1, plot=True, z_ap=FOCUS[2]):

    alph = alpha
    if (starting_rays is None):
        theta = np.linspace(np.pi / 2 - theta_bound, np.pi /
                            2 + theta_bound, n_linear)
        phi = np.linspace(np.pi / 2 - theta_bound, np.pi /
                          2 + theta_bound, n_linear)

        theta, phi = np.meshgrid(theta, phi)
        theta = np.ravel(theta)
        phi = np.ravel(phi)

        # Step 2: calculate the position + local surface normal for the dish
        n_pts = len(theta)
    else:
        n_pts = len(starting_rays)

    out = [[], [], [], [], [], [], [], [], []]

    for ii in range(n_pts):

        if (starting_rays is None):
            th = theta[ii]
            ph = phi[ii]

            # Direction of ray away from the starting point
            r_hat = [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                     np.cos(th)]

            alpha = r_hat[0]
            beta = r_hat[1]
            gamma = r_hat[2]

            # I refer to the starting point of the rays as 'rx' throughout the
            # code.  So when you see that, that's what that means.
            x_0 = P_rx[0]
            y_0 = P_rx[1]
            z_0 = P_rx[2]
        else:
            x_0, y_0, z_0 = starting_rays[ii][2]
            alpha, beta, gamma = starting_rays[ii][3]

        # print('P_rx: %s' % P_rx)
        # print('r_hat: %s' % r_hat)

        # To hit a surface, I use a root finder. This tells the ray to search
        # for a distance 't' steps away until it hits a defined surface. The
        # variable 't' is basically how many normal vectors do I need to travel
        # before I hit the desired surface. I do this for every surface
        # throughout this function.

        def root_11(t):

            x = x_0 + alpha * t  # the variable 't' I explained above ^
            y = y_0 + beta * t
            z = z_0 + gamma * t

            x11, y11, z11 = ref_tele_to_11(
                x, y, z, 1
            )  # Convert ray's endpoint into the surface's reference frame

            z_11 = l1s1(x11, y11)  # Z of surface in surface's reference frame
            # I just put this in here to make sure nothing crazy happens
            # (i.e. avoid nan values)
            if np.isnan(z_11) == True:
                z_11 = 0
            root = z11 - z_11
            return root

        try:
            t_11 = optimize.brentq(root_11, 1e-3, 90)
        except ValueError:
            continue

        # Location of where ray hits lens 1 surface 1
        x_11 = x_0 + alpha * t_11
        y_11 = y_0 + beta * t_11
        z_11 = z_0 + gamma * t_11
        P_11 = np.array([x_11, y_11, z_11])
        # print('P_11: %s' % P_11)

        ###### Transform into lens 1 surface 1 coordinates ##########################
        x_11_temp, y_11_temp, z_11_temp = ref_tele_to_11(
            x_11, y_11, z_11, 1
        )
        x_rx_temp, y_rx_temp, z_rx_temp = ref_tele_to_11(
            x_0, y_0, z_0, 1)  # P_rx temp
        norm = d_z(x_11_temp, y_11_temp, lens1_surf1, "spheric")
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))

        vec_rx_11 = np.array([x_11_temp, y_11_temp, z_11_temp]) - np.array(
            [x_rx_temp, y_rx_temp, z_rx_temp]
        )
        dist_rx_11 = np.sqrt(np.sum(vec_rx_11 ** 2))
        tan_rx_11 = vec_rx_11 / dist_rx_11

        # Use Snell's Law to find angle of outgoing ray:
        tan_og_lens = snell_vec(n_vac, n_lens, -N_hat, tan_rx_11)

        ###### Transform back into telescope reference frame ##########################
        N_hat_t_x, N_hat_t_y, N_hat_t_z = ref_11_to_tele(
            N_hat[0], N_hat[1], N_hat[2], 0
        )
        N_hat_t = [-N_hat_t_x, -N_hat_t_y, -N_hat_t_z]
        tan_rx_11_t_x, tan_rx_11_t_y, tan_rx_11_t_z = ref_11_to_tele(
            tan_rx_11[0], tan_rx_11[1], tan_rx_11[2], 0
        )
        tan_rx_11_t = [tan_rx_11_t_x, tan_rx_11_t_y, tan_rx_11_t_z]
        tan_og_t_x, tan_og_t_y, tan_og_t_z = ref_11_to_tele(
            tan_og_lens[0], tan_og_lens[1], tan_og_lens[2], 0
        )
        tan_og_t = [tan_og_t_x, tan_og_t_y, tan_og_t_z]

        # I've commented these out, but if you want to view the outgoing ray direction,
        # normal vector of surface, incoming ray direction, and  of this surface
        # you can uncomment and plot the normal vector of the 13th ray.
        #         if ii==13:
        #             plt.plot([y_11,y_11+(5*tan_og_t[1])], [z_11,z_11+(5*tan_og_t[2])], "-",color = 'r')
        #             plt.plot([y_11,y_11+(5*N_hat_t[1])], [z_11,z_11+(5*N_hat_t[2])], "-",color = 'k')
        #             plt.plot([y_11,y_11-(5*tan_rx_11_t[1])], [z_11,z_11-(5*tan_rx_11_t[2])], "-",color = 'g')

        # This is the direction of the outgoing ray from
        # lens 1 surface 1 towards lens 1 surface 2:
        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        # Now we repeat the above, but we search for where
        # the ray hits lens 1 surface 2... etc...
        def root_12(t):

            x = x_11 + alpha * t
            y = y_11 + beta * t
            z = z_11 + gamma * t

            x12, y12, z12 = ref_tele_to_12(
                x, y, z, 1
            )  # Convert ray's endpoint into M2 coordinates

            z_12 = l1s2(x12, y12)  # Z of mirror in M2 coordinates
            if np.isnan(z_12) == True:
                z_12 = 0
            root = z12 - z_12
            return root

        try:
            t_12 = optimize.brentq(root_12, 1e-3, 90)
        except ValueError:
            continue

        x_12 = P_11[0] + alpha * t_12
        y_12 = P_11[1] + beta * t_12
        z_12 = P_11[2] + gamma * t_12
        P_12 = np.array([x_12, y_12, z_12])

        ###### in lens surface reference frame ##########################
        x_12_temp, y_12_temp, z_12_temp = ref_tele_to_12(
            x_12, y_12, z_12, 1
        )  # P_m2 temp
        x_11_temp, y_11_temp, z_11_temp = ref_tele_to_12(
            x_11, y_11, z_11, 1
        )  # P_rx temp
        norm = d_z(x_12_temp, y_12_temp, lens1_surf2, "aspheric")
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))

        vec_11_12 = np.array([x_12_temp, y_12_temp, z_12_temp]) - np.array(
            [x_11_temp, y_11_temp, z_11_temp]
        )
        dist_11_12 = np.sqrt(np.sum(vec_11_12 ** 2))
        tan_11_12 = vec_11_12 / dist_11_12

        # Use Snell's Law to find angle of outgoing ray:

        tan_og_lens = snell_vec(n_lens, n_vac, -N_hat, tan_11_12)

        ###### Transform back into telescope reference frame ##########################
        N_hat_t_x, N_hat_t_y, N_hat_t_z = ref_12_to_tele(
            N_hat[0], N_hat[1], N_hat[2], 0
        )
        N_hat_t = [N_hat_t_x, N_hat_t_y, N_hat_t_z]
        tan_11_12_t_x, tan_11_12_t_y, tan_11_12_t_z = ref_12_to_tele(
            tan_11_12[0], tan_11_12[1], tan_11_12[2], 0
        )
        tan_11_12_t = [tan_11_12_t_x, tan_11_12_t_y, tan_11_12_t_z]
        tan_og_t_x, tan_og_t_y, tan_og_t_z = ref_12_to_tele(
            tan_og_lens[0], tan_og_lens[1], tan_og_lens[2], 0
        )
        tan_og_t = [tan_og_t_x, tan_og_t_y, tan_og_t_z]

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        def root_21(t):
            x = P_12[0] + alpha * t
            y = P_12[1] + beta * t
            z = P_12[2] + gamma * t

            x21, y21, z21 = ref_tele_to_21(
                x, y, z, 1
            )  # take ray end coordinates and convert to M1 coordinates
            z_21 = l2s1(x21, y21)  # Z of mirror 1 in M1 coordinates
            if np.isnan(z_21) == True:
                z_21 = 0
            root = z21 - z_21
            return root

        try:
            t_21 = optimize.brentq(root_21, 2, 30)
        except ValueError:
            continue

        # Location of where ray hits M1
        x_21 = P_12[0] + alpha * t_21
        y_21 = P_12[1] + beta * t_21
        z_21 = P_12[2] + gamma * t_21
        P_21 = np.array([x_21, y_21, z_21])

        # print(80 * '-')

        ###### in lens surface reference frame ##########################
        x_21_temp, y_21_temp, z_21_temp = ref_tele_to_21(
            x_21, y_21, z_21, 1
        )  # P_m2 temp
        x_12_temp, y_12_temp, z_12_temp = ref_tele_to_21(
            x_12, y_12, z_12, 1
        )  # P_rx temp

        norm = d_z(x_21_temp, y_21_temp, lens2_surf1, "spheric")
        # print(f'norm at 21 = {np.around(norm, 5)}')
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))

        vec_12_21 = np.array([x_21_temp, y_21_temp, z_21_temp]) - np.array(
            [x_12_temp, y_12_temp, z_12_temp]
        )
        dist_12_21 = np.sqrt(np.sum(vec_12_21 ** 2))
        tan_12_21 = vec_12_21 / dist_12_21

        # Use Snell's Law to find angle of outgoing ray:

        # print(
        #     f'snells law at 21 to go to 22: n_lens = {n_lens}. \n N_hat = {np.around(N_hat, 5)}. tan_12_21 = {np.around(tan_12_21, 5)}')
        tan_og_lens = snell_vec(n_vac, n_lens, -N_hat, tan_12_21)
        # print(f'normal vec spit out from snells={np.around(tan_og_lens, 5)}')

        ###### Transform back into telescope reference frame ##########################
        N_hat_t_x, N_hat_t_y, N_hat_t_z = ref_21_to_tele(
            N_hat[0], N_hat[1], N_hat[2], 0
        )
        N_hat_t = [N_hat_t_x, N_hat_t_y, N_hat_t_z]
        tan_12_21_t_x, tan_12_21_t_y, tan_12_21_t_z = ref_21_to_tele(
            tan_12_21[0], tan_12_21[1], tan_12_21[2], 0
        )
        tan_12_21_t = [tan_12_21_t_x, tan_12_21_t_y, tan_12_21_t_z]
        tan_og_t_x, tan_og_t_y, tan_og_t_z = ref_21_to_tele(
            tan_og_lens[0], tan_og_lens[1], tan_og_lens[2], 0
        )
        tan_og_t = [tan_og_t_x, tan_og_t_y, tan_og_t_z]

        # print(
        #     f'normal vector from 21 to 22 forwards in global frame = {np.around(tan_og_t, decimals=5)}')

        # print(
        #     f'intersection at point 21 going forwards= {np.around(P_21, decimals=5)}')
        # print(
        #     f'normal vector going from 12 to 21 forwards = {np.around(tan_12_21, decimals=5)}')

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        def root_22(t):
            x = P_21[0] + alpha * t
            y = P_21[1] + beta * t
            z = P_21[2] + gamma * t

            x22, y22, z22 = ref_tele_to_22(
                x, y, z, 1
            )
            z_22 = l2s2(x22, y22)
            if np.isnan(z_22) == True:
                z_22 = 0
            root = z22 - z_22
            return root

        try:
            t_22 = optimize.brentq(root_22, 0.05, 10)
        except ValueError:
            continue

        # Location of where ray hits lens surface
        x_22 = P_21[0] + alpha * t_22
        y_22 = P_21[1] + beta * t_22
        z_22 = P_21[2] + gamma * t_22
        P_22 = np.array([x_22, y_22, z_22])

        # print(f'P21 intersect = {np.around(P_21, decimals=5)}')
        # print(f'initial normal vec = {np.around(tan_og_t, decimals=5)}')
        # print(f'P22 intersect = {np.around(P_22, decimals=5)}')
        # print(80 * '-')

        # print(
        #     f'intersect at lens 22 going forwards = {np.around(P_22, decimals=5)}')

        ###### in lens surface reference frame ##########################
        x_22_temp, y_22_temp, z_22_temp = ref_tele_to_22(
            x_22, y_22, z_22, 1
        )  # P_m2 temp
        x_21_temp, y_21_temp, z_21_temp = ref_tele_to_22(
            x_21, y_21, z_21, 1
        )  # P_rx temp

        # print(
        #     f'vec at 21 in 22 surface frame = {np.around([x_21_temp, y_21_temp, z_21_temp], decimals=5)}')
        # print(
        #     f'vec at 22 in 22 surface frame = {np.around([x_22_temp, y_22_temp, z_22_temp], decimals=5)}')
        norm = d_z(x_22_temp, y_22_temp, lens2_surf2, "spheric")
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))

        vec_21_22 = np.array([x_22_temp, y_22_temp, z_22_temp]) - np.array(
            [x_21_temp, y_21_temp, z_21_temp]
        )
        dist_21_22 = np.sqrt(np.sum(vec_21_22 ** 2))

        tan_21_22 = vec_21_22 / dist_21_22

        # Use Snell's Law to find angle of outgoing ray:

        tan_og_lens = snell_vec(n_lens, n_vac, -N_hat, tan_21_22)

        ###### Transform back into telescope reference frame ##########################
        N_hat_t_x, N_hat_t_y, N_hat_t_z = ref_22_to_tele(
            N_hat[0], N_hat[1], N_hat[2], 0
        )
        N_hat_t = [N_hat_t_x, N_hat_t_y, N_hat_t_z]
        tan_21_22_t_x, tan_21_22_t_y, tan_21_22_t_z = ref_22_to_tele(
            tan_21_22[0], tan_21_22[1], tan_21_22[2], 0
        )
        tan_21_22_t = [tan_21_22_t_x, tan_21_22_t_y, tan_21_22_t_z]
        tan_og_t_x, tan_og_t_y, tan_og_t_z = ref_22_to_tele(
            tan_og_lens[0], tan_og_lens[1], tan_og_lens[2], 0
        )
        tan_og_t = [tan_og_t_x, tan_og_t_y, tan_og_t_z]

        # print(
        #     f'normal vector from 21 to 22 forwards = {np.around(tan_21_22, decimals=5)}')

        ##################################################
        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        def root_flat(t):

            x = x_22 + alpha * t
            y = y_22 + beta * t
            z = z_22 + gamma * t
            xflat, yflat, zflat = ref_tele_to_flat(
                x, y, z, 1
            )
            z_flat = flat(xflat, yflat)
            if np.isnan(z_flat) == True:
                z_flat = 0
            root = zflat - z_flat
            return root

        try:
            t_flat = optimize.brentq(root_flat, 0.03, 50)
        except ValueError:
            continue

        # Location of where ray hits M2
        x_flat = x_22 + alpha * t_flat
        y_flat = y_22 + beta * t_flat
        z_flat = z_22 + gamma * t_flat
        P_flat = np.array([x_flat, y_flat, z_flat])

        ###### in lens surface reference frame ##########################
        x_flat_temp, y_flat_temp, z_flat_temp = ref_tele_to_flat(
            x_flat, y_flat, z_flat, 1
        )  # P_m2 temp
        x_22_temp, y_22_temp, z_22_temp = ref_tele_to_flat(
            x_22, y_22, z_22, 1
        )  # P_rx temp
        norm_temp = np.array([0, 0, 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))

        vec_22_flat = np.array([x_flat_temp, y_flat_temp, z_flat_temp]) - np.array(
            [x_22_temp, y_22_temp, z_22_temp]
        )
        dist_22_flat = np.sqrt(np.sum(vec_22_flat ** 2))
        tan_22_flat = vec_22_flat / dist_22_flat

        tan_og_lens = 2 * (np.dot(N_hat, -tan_22_flat)
                           * (N_hat)) - (-tan_22_flat)
        ###### Transform back into telescope reference frame ##########################
        N_hat_t_x, N_hat_t_y, N_hat_t_z = ref_flat_to_tele(
            N_hat[0], N_hat[1], N_hat[2], 0
        )
        N_hat_t = [N_hat_t_x, N_hat_t_y, N_hat_t_z]
        tan_22_flat_t_x, tan_22_flat_t_y, tan_22_flat_t_z = ref_flat_to_tele(
            tan_22_flat[0], tan_22_flat[1], tan_22_flat[2], 0
        )
        tan_22_flat_t = [tan_22_flat_t_x, tan_22_flat_t_y, tan_22_flat_t_z]
        tan_og_t_x, tan_og_t_y, tan_og_t_z = ref_flat_to_tele(
            tan_og_lens[0], tan_og_lens[1], tan_og_lens[2], 0
        )
        tan_og_t = [tan_og_t_x, tan_og_t_y, tan_og_t_z]

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        def root_31(t):
            x = x_flat + alpha * t
            y = y_flat + beta * t
            z = z_flat + gamma * t
            x31, y31, z31 = ref_tele_to_31(
                x, y, z, 1
            )
            z_31 = l3s1(x31, y31)
            if np.isnan(z_31) == True:
                z_31 = 0
            root = z31 - z_31
            return root

        try:
            t_31 = optimize.brentq(root_31, 0.01, 500)
        except ValueError:
            continue
        # Location of where ray hits M1
        x_31 = P_flat[0] + alpha * t_31
        y_31 = P_flat[1] + beta * t_31
        z_31 = P_flat[2] + gamma * t_31
        P_31 = np.array([x_31, y_31, z_31])

        ###### in lens surface reference frame ##########################
        x_31_temp, y_31_temp, z_31_temp = ref_tele_to_31(
            x_31, y_31, z_31, 1
        )  # P_m2 temp
        x_flat_temp, y_flat_temp, z_flat_temp = ref_tele_to_31(
            x_flat, y_flat, z_flat, 1
        )  # P_rx temp
        norm = d_z(x_31_temp, y_31_temp, lens3_surf1, "aspheric")
        norm_temp = np.array([-norm[0], -norm[1], 1])

        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))
        vec_flat_31 = np.array([x_31_temp, y_31_temp, z_31_temp]) - np.array(
            [x_flat_temp, y_flat_temp, z_flat_temp]
        )
        dist_flat_31 = np.sqrt(np.sum(vec_flat_31 ** 2))
        tan_flat_31 = vec_flat_31 / dist_flat_31

        tan_og_lens = snell_vec(n_vac, n_lens, N_hat, tan_flat_31)

        ###### Transform back into telescope reference frame ##########################
        N_hat_t_x, N_hat_t_y, N_hat_t_z = ref_31_to_tele(
            N_hat[0], N_hat[1], N_hat[2], 0
        )
        N_hat_t = [N_hat_t_x, N_hat_t_y, N_hat_t_z]
        tan_flat_31_t_x, tan_flat_31_t_y, tan_flat_31_t_z = ref_31_to_tele(
            tan_flat_31[0], tan_flat_31[1], tan_flat_31[2], 0
        )
        tan_flat_31_t = [tan_flat_31_t_x, tan_flat_31_t_y, tan_flat_31_t_z]
        tan_og_t_x, tan_og_t_y, tan_og_t_z = ref_31_to_tele(
            tan_og_lens[0], tan_og_lens[1], tan_og_lens[2], 0
        )
        tan_og_t = [tan_og_t_x, tan_og_t_y, tan_og_t_z]

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        def root_32(t):
            x = x_31 + alpha * t
            y = y_31 + beta * t
            z = z_31 + gamma * t
            x32, y32, z32 = ref_tele_to_32(
                x, y, z, 1
            )
            z_32 = l3s2(x32, y32)
            if np.isnan(z_31) == True:
                z_32 = 0
            root = z32 - z_32
            return root

        try:
            t_32 = optimize.brentq(root_32, 0.01, 500)
        except ValueError:
            continue
        # Location of where ray hits M1
        x_32 = P_31[0] + alpha * t_32
        y_32 = P_31[1] + beta * t_32
        z_32 = P_31[2] + gamma * t_32
        P_32 = np.array([x_32, y_32, z_32])

        ###### in lens surface reference frame ##########################
        x_32_temp, y_32_temp, z_32_temp = ref_tele_to_32(
            x_32, y_32, z_32, 1
        )  # P_m2 temp
        x_31_temp, y_31_temp, z_31_temp = ref_tele_to_32(
            x_31, y_31, z_31, 1
        )  # P_rx temp
        norm = d_z(x_32_temp, y_32_temp, lens3_surf2, "aspheric")
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))

        vec_31_32 = np.array([x_32_temp, y_32_temp, z_32_temp]) - np.array(
            [x_31_temp, y_31_temp, z_31_temp]
        )
        dist_31_32 = np.sqrt(np.sum(vec_31_32 ** 2))
        tan_31_32 = vec_31_32 / dist_31_32

        tan_og_lens = snell_vec(n_lens, n_vac, -N_hat, tan_31_32)

        ###### Transform back into telescope reference frame ##########################
        N_hat_t_x, N_hat_t_y, N_hat_t_z = ref_32_to_tele(
            N_hat[0], N_hat[1], N_hat[2], 0
        )
        N_hat_t = [N_hat_t_x, N_hat_t_y, N_hat_t_z]
        tan_31_32_t_x, tan_31_32_t_y, tan_31_32_t_z = ref_32_to_tele(
            tan_31_32[0], tan_31_32[1], tan_31_32[2], 0
        )
        tan_31_32_t = [tan_31_32_t_x, tan_31_32_t_y, tan_31_32_t_z]
        tan_og_t_x, tan_og_t_y, tan_og_t_z = ref_32_to_tele(
            tan_og_lens[0], tan_og_lens[1], tan_og_lens[2], 0
        )

        tan_og_t = [tan_og_t_x, tan_og_t_y, tan_og_t_z]

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        dist_32_ap = abs((z_ap - P_32[2]) / tan_og_t[2])

        total_path_length = (
            t_11 + t_12 * n_lens + t_21 + t_22 * n_lens +
            t_flat + t_31 + t_32 * n_lens + dist_32_ap
        )

        # print(t_11, t_12, t_21, t_22, t_flat, t_31,
        #       t_32, dist_32_ap)

        if (starting_rays is not None):
            # convert to mm
            total_path_length = (total_path_length / mm_to_in) + (
                starting_rays[ii][4])
        pos_ap = P_32 + dist_32_ap * np.array(tan_og_t)

        # Estimate theta, phi of outgoing initial ray
        de_ve = np.arctan(tan_rx_11_t[2] / (-tan_rx_11_t[1]))
        de_ho = np.arctan(
            tan_rx_11_t[0] / np.sqrt(tan_rx_11_t[2] ** 2 + tan_rx_11_t[1] ** 2)
        )

        #################################################
        ## Plotting lenses and rays #####################
        x_points = [x_0, x_11, x_12, x_21, x_22, x_flat, x_31, x_32, pos_ap[0]]
        y_points = [y_0, y_11, y_12, y_21, y_22, y_flat, y_31, y_32, pos_ap[1]]
        z_points = [z_0, z_11, z_12, z_21, z_22, z_flat, z_31, z_32, pos_ap[2]]

        # These rays aren't going to hit the FTS and are really incorrect here
        # (surfaces defined don't really have boundaries so weird things can
        # happen), so we discard them
        # if np.abs(pos_ap[0]) > 2 or np.abs(pos_ap[2]) > 2:
        #     continue
        if (plot):
            for i in range(len(x_points) - 1):
                plot_ray([x_points[i], x_points[i + 1]],
                         [y_points[i], y_points[i + 1]],
                         [z_points[i], z_points[i + 1]], fig, col, alph,
                         shift=shift_origin, tilt=tilt_angle,
                         fac=(1 / mm_to_in))

        if (starting_rays is None):
            pol_angle = 0
            intensity = 1
        else:
            pol_angle = starting_rays[ii][0]
            intensity = starting_rays[ii][1]
        # Write out
        out[0].append(pos_ap[0])
        out[1].append(pos_ap[1])
        out[2].append(pos_ap[2])

#         out[4, ii] = np.exp(
#             (-0.5)
#             * ((de_ve) ** 2 + (de_ho) ** 2)
#             / ((24 * np.pi / 180) / (np.sqrt(8 * np.log(2)))) ** 2 #amplitude
       # ) # Setting FWHP = 24 degrees, you should change this

        # out[5].append(N_hat_t[0])
        # out[6].append(N_hat_t[1])
        # out[7].append(N_hat_t[2])

        out[3].append(tan_og_t[0])
        out[4].append(tan_og_t[1])
        out[5].append(tan_og_t[2])

        out[6].append(pol_angle)
        out[7].append(intensity)
        out[8].append(total_path_length)  # phase

    if (plot):
        plot_surface(Xt11, Yt11, Zt11, fig, 0, 100, 'teal', half_index=50,
                     shift=shift_origin, tilt=tilt_angle, fac=(1 / mm_to_in))
        plot_surface(Xt12, Yt12, Zt12, fig, 0, 100, 'teal',
                     shift=shift_origin, tilt=tilt_angle, fac=(1 / mm_to_in))

        plot_surface(Xt21, Yt21, Zt21, fig, 0, 100, 'teal',
                     shift=shift_origin, tilt=tilt_angle, fac=(1 / mm_to_in))
        plot_surface(Xt22, Yt22, Zt22, fig, 0, 100, 'teal',
                     shift=shift_origin, tilt=tilt_angle, fac=(1 / mm_to_in))

        plot_surface(Xtflat, Ytflat, Ztflat, fig, 0, 100, 'silver',
                     half_index=100, shift=shift_origin, tilt=tilt_angle,
                     fac=(1 / mm_to_in), alpha=.3)

        plot_surface(Xt31, Yt31, Zt31, fig, 0, 100, 'teal', half_index=100,
                     shift=shift_origin, tilt=tilt_angle, fac=(1 / mm_to_in))
        plot_surface(Xt32, Yt32, Zt32, fig, 0, 100, 'teal', half_index=100,
                     shift=shift_origin, tilt=tilt_angle, fac=(1 / mm_to_in))

    return np.array(out)  # , [x_points, y_points, z_points]


def run_rays_forwards(shift, n_linear, theta_bound=.3, z_ap=FOCUS[2], plot=False,
                      fig=None, color=None):
    assert z_ap <= 0
    # we want the shift in mm so now let's convert to inches
    shift = np.multiply(shift, mm_to_in)
    start_position = [0, -520 * (mm_to_in), 0]
    new_start = np.add(start_position, shift)

    out = run_rays_through_coupling_optics_forwards(
        new_start, fts_geo, 'black', fig, n_linear=n_linear,
        theta_bound=theta_bound, plot=plot, z_ap=z_ap, alpha=.2)
    # print(out)

    return out


def run_rays_forwards_input_rays(starting_rays, z_ap=FOCUS[2], plot=False,
                                 fig=None, color=None):
    assert z_ap <= 0
    # we want the shift in mm so now let's convert to inches
    # start_position = [0, -4.26 * (mm_to_in), 0]

    out = run_rays_through_coupling_optics_forwards(
        None, fts_geo, color, fig, plot=plot, z_ap=z_ap, alpha=.2,
        starting_rays=starting_rays)

    return out
