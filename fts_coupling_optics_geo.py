import numpy as np

mm_to_in = 0.03937008


class fts_geo:
    # number of rays in a 1D scan (i.e. the resolution of your beam)
    N_scan = 10


class lens1_surf1:
    c = 0.1176
    k = 0
    a_4 = 0
    diam = 4.292


class lens1_surf2:
    c = -0.2222
    k = -1.8567
    a_4 = 0.0005
    diam = 4.298


class lens2_surf1:
    c = -0.3571
    k = 0
    a_4 = 0
    diam = 1.559


class lens2_surf2:
    c = -0.4878
    k = 0
    a_4 = 0
    diam = 2.329


class lens3_surf1:
    c = -0.2260
    k = -0.9603
    a_4 = 0.0005
    diam = 5.112


class lens3_surf2:
    c = -0.0913
    k = -6.5390
    a_4 = -0.0004
    diam = 5.089


def l1s1(x, y):
    amp = z(x, y, lens1_surf1, 'spheric')
    amp = np.where(x**2 + y**2 <= (lens1_surf1.diam/2)**2, amp, 0)
    amp = np.where(x**2 + y**2 <= (lens1_surf1.diam/2)**2, amp, np.max(amp))
    return amp


def l1s2(x, y):
    amp = z(x, y, lens1_surf2, 'aspheric')
    amp = np.where(x**2 + y**2 <= (lens1_surf2.diam/2)**2, amp, 0)
    amp = np.where(x**2 + y**2 <= (lens1_surf2.diam/2)**2, amp, np.min(amp))
    return amp


def l2s1(x, y):
    amp = z(x, y, lens2_surf1, 'spheric')
    amp = np.where(x**2 + y**2 <= (lens2_surf1.diam/2)**2, amp, 0)
    amp = np.where(x**2 + y**2 <= (lens2_surf1.diam/2)**2, amp, np.min(amp))
    return amp


def l2s2(x, y):
    amp = z(x, y, lens2_surf2, 'spheric')
    amp = np.where(x**2 + y**2 <= (lens2_surf2.diam/2)**2, amp, 0)
    amp = np.where(x**2 + y**2 <= (lens2_surf2.diam/2)**2, amp, np.min(amp))
    return amp


def l3s1(x, y):
    amp = z(x, y, lens3_surf1, 'aspheric')
    amp = np.where(x**2 + y**2 <= (lens3_surf1.diam/2)**2, amp, 0)
    amp = np.where(x**2 + y**2 <= (lens3_surf1.diam/2)**2, amp, np.min(amp))
    return amp


def l3s2(x, y):
    amp = z(x, y, lens3_surf2, 'aspheric')
    amp = np.where(x**2 + y**2 <= (lens3_surf2.diam/2)**2, amp, 0)
    amp = np.where(x**2 + y**2 <= (lens3_surf2.diam/2)**2, amp, np.min(amp))
    return amp


###### Lens 1 #############
def ref_tele_to_11(x, y, z, shift):
    if shift == 1:

        y += (.276 + (.15))
    x_temp = x
    y_temp = y * np.cos(np.pi/2) - z * np.sin(np.pi/2)
    z_temp = y * np.sin(np.pi/2) + z * np.cos(np.pi/2)
    return x_temp, y_temp, z_temp


def ref_11_to_tele(x, y, z, shift):

    x_temp = x
    y_temp = y * np.cos(-np.pi/2) - z * np.sin(-np.pi/2)
    z_temp = y * np.sin(-np.pi/2) + z * np.cos(-np.pi/2)

    if shift == 1:
        y_temp -= (.276 + (.15))
    return x_temp, y_temp, z_temp


def ref_tele_to_12(x, y, z, shift):
    if shift == 1:
        y -= (.501)
    x_temp = x
    y_temp = y * np.cos(np.pi/2) - z * np.sin(np.pi/2)
    z_temp = y * np.sin(np.pi/2) + z * np.cos(np.pi/2)
    return x_temp, y_temp, z_temp


def ref_12_to_tele(x, y, z, shift):
    x_temp = x
    y_temp = y * np.cos(-np.pi/2) - z * np.sin(-np.pi/2)
    z_temp = y * np.sin(-np.pi/2) + z * np.cos(-np.pi/2)
    if shift == 1:
        y_temp += (.501)
    return x_temp, y_temp, z_temp

###### Lens 2 #############


def ref_tele_to_21(x, y, z, shift):
    if shift == 1:
        y -= ((210*mm_to_in))
    x_temp = x
    y_temp = y * np.cos(np.pi/2) - z * np.sin(np.pi/2)
    z_temp = y * np.sin(np.pi/2) + z * np.cos(np.pi/2)
    return x_temp, y_temp, z_temp


def ref_21_to_tele(x, y, z, shift):
    x_temp = x
    y_temp = y * np.cos(-np.pi/2) - z * np.sin(-np.pi/2)
    z_temp = y * np.sin(-np.pi/2) + z * np.cos(-np.pi/2)
    if shift == 1:
        y_temp += ((210*mm_to_in))
    return x_temp, y_temp, z_temp


def ref_tele_to_22(x, y, z, shift):
    if shift == 1:
        y -= (1.848+.363 + (210*mm_to_in))
    x_temp = x
    y_temp = y * np.cos(np.pi/2) - z * np.sin(np.pi/2)
    z_temp = y * np.sin(np.pi/2) + z * np.cos(np.pi/2)
    return x_temp, y_temp, z_temp


def ref_22_to_tele(x, y, z, shift):

    x_temp = x
    y_temp = y * np.cos(-np.pi/2) - z * np.sin(-np.pi/2)
    z_temp = y * np.sin(-np.pi/2) + z * np.cos(-np.pi/2)
    if shift == 1:
        y_temp += (1.848+.363 + (210*mm_to_in))
    return x_temp, y_temp, z_temp

###### Flat Reflector #############


def ref_tele_to_flat(x, y, z, shift):
    if shift == 1:
        y -= +((210+82)*mm_to_in + 1.848)
    x_temp = x
    y_temp = y * np.cos(np.pi/4) - z * np.sin(np.pi/4)
    z_temp = y * np.sin(np.pi/4) + z * np.cos(np.pi/4)
    return x_temp, y_temp, z_temp


def ref_flat_to_tele(x, y, z, shift):
    x_temp = x
    y_temp = y * np.cos(-np.pi/4) - z * np.sin(-np.pi/4)
    z_temp = y * np.sin(-np.pi/4) + z * np.cos(-np.pi/4)
    if shift == 1:
        y_temp += ((210+82)*mm_to_in + 1.848)
    return x_temp, y_temp, z_temp

###### Lens 3 #############


def ref_tele_to_31(x, y, z, shift):
    if shift == 1:

        y -= +((210+82)*mm_to_in + 1.848)
        z -= (-(280*mm_to_in)+.15+.762)
    x_temp = x
    y_temp = y * np.cos(0) - z * np.sin(0)
    z_temp = y * np.sin(0) + z * np.cos(0)
    return x_temp, y_temp, z_temp


def ref_31_to_tele(x, y, z, shift):

    x_temp = x
    y_temp = y * np.cos(0) - z * np.sin(0)
    z_temp = y * np.sin(0) + z * np.cos(0)
    if shift == 1:
        y_temp += ((210+82)*mm_to_in + 1.848)
        z_temp += (-(280*mm_to_in)+.15+.762)
    return x_temp, y_temp, z_temp


def ref_tele_to_32(x, y, z, shift):
    if shift == 1:

        y -= ((210+82)*mm_to_in + 1.848)
        z -= (-(280*mm_to_in)-.255)
    x_temp = x
    y_temp = y * np.cos(np.pi) - z * np.sin(np.pi)
    z_temp = y * np.sin(np.pi) + z * np.cos(np.pi)
    return x_temp, y_temp, z_temp


def ref_32_to_tele(x, y, z, shift):

    x_temp = x
    y_temp = y * np.cos(np.pi) - z * np.sin(np.pi)
    z_temp = y * np.sin(np.pi) + z * np.cos(np.pi)
    if shift == 1:
        y_temp += ((210+82)*mm_to_in + 1.848)
        z_temp += (-(280*mm_to_in)-.255)
    return x_temp, y_temp, z_temp

# Here I define the equations of the surfaces of the lenses as the general equation provided.
# To get a specific lens surfaces, the corresponding class of variables is passed in as 'params'
# which tells the function z() exactly how to shape the surface.


def z(x, y, params, shape):
    r = np.sqrt(x**2 + y**2)

    c = params.c
    k = params.k
    a_4 = params.a_4

    if shape == 'aspheric':
        amp = (c*r**2) / (1+np.sqrt(1-((1+k)*c**2 * r**2)))
        amp += a_4*r**4
    elif shape == 'spheric':
        amp = (c*r**2) / (1+np.sqrt(1-(c**2 * r**2)))
    return amp

# This is a simple funcion defining a plane, which we will rotate into the proper position
# and this will serve as our reflecting surface (between lenses 2 and 3):


def flat(x, y):
    amp = 0
    return amp

# I have taken the derivative of z(r) by hand, and written in this code. Given an x and y position
# this function determines the x and y componants of the normal vector on that surface.


def d_z(x, y, params, shape):

    r = np.sqrt(x**2 + y**2)

    c = params.c
    k = params.k
    a_4 = params.a_4

    if shape == 'aspheric':
        coeff_1 = (a_4*4*r**2)
        coeff_2 = (c*2) / (1+np.sqrt(1-((1+k)*c**2 * r**2)))
        coeff_3 = (c**3 * (k+1)*r**2) / (np.sqrt(1-((1+k)*c**2 * r**2))
                                         * (1+np.sqrt(1-((1+k)*c**2 * r**2)))**2)
        amp_x = x*(coeff_1 + coeff_2 + coeff_3)
        amp_y = y*(coeff_1 + coeff_2 + coeff_3)
    elif shape == 'spheric':
        coeff_1 = 0
        k = 0
        coeff_2 = (c*2) / (1+np.sqrt(1-((1+k)*c**2 * r**2)))
        coeff_3 = (c**3 * (k+1)*r**2) / (np.sqrt(1-((1+k)*c**2 * r**2))
                                         * (1+np.sqrt(1-((1+k)*c**2 * r**2)))**2)
        amp_x = x*(coeff_1 + coeff_2 + coeff_3)
        amp_y = y*(coeff_1 + coeff_2 + coeff_3)

    if (np.sqrt(x**2 + y**2) >= (params.diam/2)):
        amp_x = 0
        amp_y = 0

    return amp_x, amp_y
