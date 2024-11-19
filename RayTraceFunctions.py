'''This includes the functions of several different simulations of the Meyer
Lab's Compact Fourier Transform Spectrometer. It ONLY includes the functions
that are necessary FOR THE FINAL SIMULATIONS, rather than ones that were used
when building this. It also includes different version of functions regarding
input rays (random initial phase or all zero) and pickling (the ability to
return every single ray generated in the simulation). If you are interested in
more functions showing the build up of this simulation, contact me at
liusarkarm@uchicago.edu Mira Liu'''
import yaml
from tqdm import tqdm
import time
import math
import random
from random import uniform
import numpy
import numpy as np
from numba import jit, njit

# from numba import jot  # for speedups


# Create new functions to set each of the origins and angles:
def get_aspect(config, aspect, element, number):
    '''Obtain the config for a given aspect, element, and number.

    Parameters:
        config (yaml file)  -- yaml configuration file loaded
        aspect (str)        -- aspect of the FTS (origins, angles,
                               coefficients, etc)

        element (str)       -- element of the FTS for which this aspect
                               is defined (ellipses, mirror, polarizers)

        number (int)        -- number of the element that we're specifically
                               interested in (1-10 for ellipses, 1-4 for
                               polarizers)'''

    def get_item(dic, key, num):
        if type(dic[key]) is dict:
            return dic[key][num]
        else:
            return dic[key]

    if element is None:
        return get_item(config, aspect, number)
    else:
        return get_item(config[aspect], element, number)


filename = "lab_fts_dims_mcmahon.yml"
# filename = "lab_fts_dims_dicts.yml"
# filename = "lab_fts_dims_dicts_old.yml"
with open(filename, "r") as stream:
    config_to_use = yaml.safe_load(stream)

originG = [0., 0., 0.]  # the global origin
thetG = [0., 0., 0.]  # rotation with respect to itself aka 0,0,0

p1, p2, p3, p4 = [get_aspect(config_to_use, 'polarizer_values', None,
                             i + 1) for i in range(4)]
origin1, origin2, origin3, origin4, origin5, origin6, origin7, origin8, \
    origin9, origin10 = [get_aspect(config_to_use, 'origins', 'ellipses',
                                    i + 1) for i in range(10)]

coeffellipse7 = get_aspect(config_to_use, 'coefficients', 'ellipses', 7)
coeffellipse56 = get_aspect(config_to_use, 'coefficients', 'ellipses', 5)
coeffellipse = get_aspect(config_to_use, 'coefficients', 'ellipses', 1)

thet, thet5, thet6, thet7, thet10 = [get_aspect(
    config_to_use, 'angles', 'ellipses', i) for i in [1, 5, 6, 7, 10]]

# Locations and coefficients of the polarizers:
coeffmirr = get_aspect(config_to_use, 'coefficients', 'mirror', None)
coeffpolar = get_aspect(config_to_use, 'coefficients', 'polarizers', None)

originpolar1, originpolar2, originpolar3, originpolar4 = [get_aspect(
    config_to_use, 'origins', 'polarizers', i + 1) for i in range(4)]

center1, center2, center3, center4, center5, center6, center7, center8, \
    center9, center10 = [get_aspect(
        config_to_use, 'centers', None, i + 1) for i in range(10)]

range1, range2, range3, range4, range5, range6, range7, range8, \
    range9, range10 = [get_aspect(
        config_to_use, 'ranges', None, i + 1) for i in range(10)]

''' Below are functions used in the simulation'''


'''Rotations: Give angle wanted rotated to respective function, returns rotated
point(s).'''


# @njit
def Rx(x):
    # Rx = np.matrix([[1, 0, 0], [0, np.cos(x), -np.sin(x)],
    #                 [0, np.sin(x), np.cos(x)]])
    Rx = np.array([[1., 0., 0.], [0., np.cos(x), -np.sin(x)],
                   [0., np.sin(x), np.cos(x)]])
    return Rx


# @njit
def Ry(y):
    # Ry = np.matrix([[np.cos(y), 0, np.sin(y)], [
    #                0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Ry = np.array([[np.cos(y), 0., np.sin(y)], [
                   0., 1., 0.], [-np.sin(y), 0., np.cos(y)]])
    return Ry


# @njit
def Rz(z):
    # Rz = np.matrix([[np.cos(z), - np.sin(z), 0],
    #                 [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    Rz = np.array([[np.cos(z), - np.sin(z), 0.],
                   [np.sin(z), np.cos(z), 0.], [0., 0., 1.]])
    return Rz


# @jit
def Rxyz(thet):
    Rxyz = Rx(thet[0]).dot(Ry(thet[1])).dot(Rz(thet[2]))
    return Rxyz


# @jit
def ELIorganize(p, v, coeffellipse):
    return p[0], p[1], p[2], v[0], v[1], v[2], coeffellipse[0], coeffellipse[1]


''' EllipseLineInt(ELI): Give point of the line, vector of the line, and
coefficients of the ellipse, find the intersection(s) of the line and the
ellipsoid (assuming ellipse is rotated about the x-axis.  This is where
(x-x_0)/a = (y-y_0)/b) = (z-z_0)/c. And, x^2/d^2 + y^2/e^2 + z^2/e^2 = 1
(ellipse is rotated around x axis to form ellipsoid'''


# @jit
def ELI2(p, v, coeffellipse):
    x0, y0, z0, a, b, c, d, e = ELIorganize(p, v, coeffellipse)
    A = (e**2)/(d**2) + (b**2)/(a**2) + (c**2)/(a**2)
    B = (-2*x0*b**2)/(a**2) + (2*y0*b)/(a) + (-2*x0*c**2)/(a**2) + (2*z0*c)/(a)
    C = ((x0**2)*(b**2))/(a**2) + (-2*y0*b*x0)/(a) + y0**2 + \
        ((x0**2)*(c**2))/(a**2) + (-2*z0*c*x0)/(a) + z0**2 - e**2
    xint = [(-B + np.sqrt((B**2) - 4*A*C))/(2*A),
            (-B - np.sqrt((B**2) - 4*A*C))/(2*A)]
    t = [(xint[0]-x0)/a, (xint[1]-x0)/a]
    yint = [y0 + t[0]*b, y0 + t[1]*b]
    zint = [z0 + t[0]*c, z0 + t[1]*c]
    return xint, yint, zint


'''NormalP: Given a point, vector, and ellipse, finds the point of intersection
and the normal of the corresponding tangent plane.'''


# @jit
def NormalP(pli, v1, coeffellipse):
    xint1, yint1, zint1 = ELI2(pli, v1, coeffellipse)
    cpos = [(2*xint1[0])/(coeffellipse[0]**2), (2*yint1[0]) /
            (coeffellipse[1]**2), (2*zint1[0])/(coeffellipse[1]**2)]
    cneg = [(2*xint1[1])/(coeffellipse[0]**2), (2*yint1[1]) /
            (coeffellipse[1]**2), (2*zint1[1])/(coeffellipse[1]**2)]
    cpos = np.array(cpos)
    cneg = np.array(cneg)
    return cpos, cneg


''' norm(N): Given a vector, returns the normal of it'''


# @jit
def N(V):
    VectL = numpy.array(V)
    VNorm = numpy.sqrt(VectL[0]**2 + VectL[1]**2 + VectL[2]**2)
    VectLNorm = ([u/VNorm for u in VectL])
    VectLNorm = numpy.array(VectLNorm)
    return VectLNorm


'''make_line (ML): given a point, vector, and length, makes the corresponding line '''

# this function is bad, don't use it.

# @jit


def ML(p, v, L):
    pointL = p
    VectL = numpy.array(v)
    Lwant = int(L)
    VectLNorm = N(v)
    t = numpy.linspace(0, Lwant, 50)  # make related to wanted length??
    x = [pointL[0]]
    y = [pointL[1]]
    z = [pointL[2]]
    for t in range(0, Lwant):
        L = numpy.sqrt(
            ((VectLNorm[0]*t)**2 + (VectLNorm[2]*t)**2 + (VectLNorm[2]*t)**2))
        xL = pointL[0] + t*VectLNorm[0]
        yL = pointL[1] + t*VectLNorm[1]
        zL = pointL[2] + t*VectLNorm[2]
        if L <= Lwant:
            x.append(xL)
            y.append(yL)
            z.append(zL)
    return x, y, z


'''setrange2d(SR2): Give radius, intersection points, and origin, only keep
points within the circle '''


# @jit
def SR2(xrange, X, Y, Z, origin):
    xintG = []
    yintG = []
    zintG = []
    for i in range(0, len(X)):
        xinti = X[i]
        yinti = Y[i]
        zinti = Z[i]
        if (xinti-origin[0])**2 + (zinti-origin[2])**2 <= xrange**2:
            xintG.append(xinti)
            yintG.append(yinti)
            zintG.append(zinti)
    return xintG, yintG, zintG


'''CreateEllipseBoundShifted(CEBS): creates an ellipse with given coefficients
at origin (0,0). Returns x, positive y, negative y, and z coordinates. (z is
assumed to be 0 as it is 2d)'''


# @jit
def CEBS(coeffellipse, length):
    xc = np.linspace(-float(length)/2, float(length) /
                     2, 100)  # centers around angle
    # pos side of ellipse
    yc1 = np.sqrt((1-(((xc)**2)/(coeffellipse[0]**2)))*coeffellipse[1]**2)
    # neg side of ellipse
    yc2 = -np.sqrt((1-(((xc)**2)/(coeffellipse[0]**2)))*coeffellipse[1]**2)
    # assumes merely in the x,y plane and uses 100 points
    zc = np.linspace(0, 0, 100)
    return xc, yc1, yc2, zc


'''RotateStrandBoundShiftCORRECTING (RSBSC): Give angle to be rotated about the
x axis, coefficients of ellipse, length to restrict ellipse, origin of
shifting, and sign(pos or neg if it is on the positive or negative side of the
y axis). Creates the ellipse rotated at the specific angle around the x-axis'''


def RSBSC(theta, coeffellipse, length, sign):
    Rotated = []
    xc, yc1, yc2, zc = CEBS(coeffellipse, length)
    if sign == 'pos':
        for i in range(0, 100):
            # number of original points using POSITIVE side of ellipse
            v = [xc[i], yc1[i], zc[i]]
            # multiplied by rotation vector
            v2 = np.array(np.dot(v, Rx(theta)))
            # take away zero-index when converting from mat -> arr
            Rotated.append(v2)  # rotated vectors
    if sign == 'neg':
        for i in range(0, 100):
            # number of original points on NEGATIVE side of ellipse
            v = [xc[i], yc2[i], zc[i]]
            # multiplied by rotation vector
            v2 = np.array(np.dot(v, Rx(theta)))
            # take away zero-index when converting from mat -> arr
            Rotated.append(v2)  # rotated vectors
    xcR1 = []
    ycR1 = []
    zcR1 = []
    for j in range(0, 100):  # recombining into arrays of x,y,z to be plotted
        xcR1.append(Rotated[j][0])
        ycR1.append(Rotated[j][1])
        zcR1.append(Rotated[j][2])
    return xcR1, ycR1, zcR1


'''CreateZBoundShiftCorrecting (CZBSC): give number of ellipses wanted, half of
the angle (theta) wanted (so if you want half of the ellipsoid, choose
np.pi/2), coefficients of ellipse, restriction length, shift origin, and sign
(pos or neg). Returns the 3d shape of the restricted and shifted ellipsoid
rotated Theta about the x-axis'''


def CZBSC(a, n, coeffellipse, length, sign):
    x1 = []
    y1 = []
    z1 = []
    for i in range(0, a):
        theta = np.linspace(0, n, a)  # range from 0 to n angles in a divisions
        x, y, z = RSBSC(theta[i], coeffellipse, length,
                        sign)  # ellipse for specific angle
        x1.extend(x)  # adding a new ellipse for each angle
        y1.extend(y)
        z1.extend(z)
    return x1, y1, z1  # returns all ellipses


'''FTSCEllipsoidCorrecting(FTSEC): give number of ellipses wanted, half of the
angle covered wanted (so if you want half of the ellipsoid, choose np.pi/2),
coefficients of ellipse, restriction length, shift origin, and sign (pos or
neg). Returns the 3d shape of the restricted and shifted ellipsoid rotated +
Theta and -Theta about the x-axis to create a symmetric ellipsoid on the pos
and neg side of the z plane.'''


def FTSEC(a, n, coeffellipse, length, sign):
    # positive side of zplane
    X, Y, Z = CZBSC(a, n, coeffellipse, length, sign)
    X1, Y1, Z1 = CZBSC(a, -n, coeffellipse, length,
                       sign)  # negative side zplane
    if sign != 'pos' and sign != 'neg':
        print('Error')
    return X, Y, Z, X1, Y1, Z1


'''Separate: given a list of points/vectors (i.e [[x1,y1,z1],[x2,y2,z2], ...] translates into a three arrays of x, y, and z values. (this is the format for the Transform function)'''


def sep(X):
    x, y, z = [], [], []
    if type(X[0]) is int or type(X[0]) is float or type(X[0]) is numpy.float64:
        x = X[0]
        y = X[1]
        z = X[2]
    else:
        for i in range(0, len(X)):
            x.append(X[i][0])
            y.append(X[i][1])
            z.append(X[i][2])
    return x, y, z


'''The reverse of sep. Translates three arrays of x,y,z values back into series
of [x,y,z] points/vectors. '''


def sepop(x, y, z):
    v = []
    if type(x) is int or type(x) is float or type(x) is numpy.float64:
        v = [x, y, z]
    else:
        for i in range(0, len(x)):
            a = [x[i], y[i], z[i]]
            v.append(a)
    return v


''' rotate (V, thetaxyz) rotates a vector about a given angle in order of (x,y,z)'''


def rotate(point, thetaxyz):
    x = point[0]
    y = point[1]
    z = point[2]
    v = [x, y, z]
    lenvect = (x**2 + y**2 + z**2)**.5
    V = N(v)
    V2 = np.array(np.dot(V, Rxyz(thetaxyz)))
    v2f = V2*lenvect  # take away zero-index when converting from mat -> arr
    return v2f


'''rotate (V, thetaxyz) rotates a vector about a given angle in order of (z,y,x)'''


def rotaterev(point, thetaxyz):
    x = point[0]
    y = point[1]
    z = point[2]
    v = [x, y, z]
    lenvect = (x**2 + y**2 + z**2)**.5
    V = N(v)
    VZ = np.array(np.dot(V, Rz(thetaxyz[2])))
    VZY = np.array(np.dot(VZ, Ry(thetaxyz[1])))
    VZYX = np.array(np.dot(VZY, Rx(thetaxyz[0])))
    v2f = VZYX*lenvect  # take away zero-index when converting from mat -> arr
    return v2f


'''given a point (or vector) and an origin (a local one in global coordinates), shifts to the Local origin in Global coordinates'''


def shift(point, origin):
    x = point[0]
    y = point[1]
    z = point[2]
    x2 = x + origin[0]
    y2 = y + origin[1]
    z2 = z + origin[2]
    v2 = [x2, y2, z2]
    return v2


'''Given a point (or three arrays of x,y,z for points), the GLOBAL coordinates are transformed to LOCAL coordinates where the LOCAL coordinates are defined in terms of the GLOBAL coordinate system through its GLOBAL origin and GLOBAL rotation. Essentially transforms point(s) from global coordinate system to given local coordinate system. the origin is the LOCAL origin in GLOBAL coordinates'''


def transformGL(x, y, z, origin, thetaxyz):
    XTR = []
    YTR = []
    ZTR = []
    if type(x) is int or type(x) is float or type(x) is numpy.float64:
        v = [x, y, z]
        if x == 0 and y == 0 and z == 0:
            vf = shift(v, negvect(origin))
            XTR = vf[0]
            YTR = vf[1]
            ZTR = vf[2]
        else:
            v2S = shift(v, negvect(origin))
            v2RS = rotaterev(v2S, negvect(thetaxyz))
            XTR = v2RS[0]
            YTR = v2RS[1]
            ZTR = v2RS[2]
    else:
        for i in range(0, len(x)):
            v = [x[i], y[i], z[i]]
            if x[i] == 0 and y[i] == 0 and z[i] == 0:
                vf = shift(v, negvect(origin))
                XTR.append(vf[0])
                YTR.append(vf[1])
                ZTR.append(vf[2])
            else:
                v2S = shift(v, negvect(origin))
                v2RS = rotaterev(v2S, negvect(thetaxyz))
                XTR.append(v2RS[0])
                YTR.append(v2RS[1])
                ZTR.append(v2RS[2])
    return XTR, YTR, ZTR


'''transforms point(s) from local coordinate system to corresponding glocal coordinate system. origin is the LOCAL origin in GLOBAL coordinates'''


def transformLG(x, y, z, origin, thetaxyz):
    XTR = []
    YTR = []
    ZTR = []
    if type(x) is int or type(x) is float or type(x) is numpy.float64:
        v = [x, y, z]
        if x == 0 and y == 0 and z == 0:
            vf = shift(v, origin)
            XTR = vf[0]
            YTR = vf[1]
            ZTR = vf[2]
        else:
            v2R = rotate(v, thetaxyz)
            v2RS = shift(v2R, origin)
            XTR = v2RS[0]
            YTR = v2RS[1]
            ZTR = v2RS[2]
    else:
        for i in range(0, len(x)):
            v = [x[i], y[i], z[i]]
            if x[i] == 0 and y[i] == 0 and z[i] == 0:
                vf = shift(v, origin)
                XTR.append(vf[0])
                YTR.append(vf[1])
                ZTR.append(vf[2])
            else:
                v2R = rotate(v, thetaxyz)
                v2RS = shift(v2R, origin)
                XTR.append(v2RS[0])
                YTR.append(v2RS[1])
                ZTR.append(v2RS[2])
    return XTR, YTR, ZTR


# given range, one point, origin, if it lies in or not
def SR3B(ranges, xinti, yinti, zinti, origin):
    xr = ranges[0]
    yr = ranges[1]
    zr = ranges[2]
    xc = origin[0]
    yc = origin[1]
    zc = origin[2]
    if (((((xinti-xc)**2)/xr**2) + (((yinti-yc)**2)/yr**2) + (((zinti-zc)**2)/zr**2))) <= 1:
        return True
    else:
        return False


'''Negvect: negates a vector. '''


def negvect(vect):
    if type(vect[0]) is int or type(vect[0]) is float or type(vect[0]) is numpy.float64:
        vectset = [-x for x in vect]
    else:
        vectset = [[-y for y in x] for x in vect]
    return vectset


''' Spec: give the number of rays wanted, returns specular distribution of n vectors. Adapted from Meyer's Specular notebook.'''


def spec(n):
    x, y, z = [], [], []
    for i in np.arange(n):
        theta = np.arccos(uniform(-1, 1))
        phi = np.random.uniform(0, 2*np.pi)
        xt = np.sin(theta)*np.cos(phi)
        yt = np.sin(theta)*np.sin(phi)
        zt = np.cos(theta)
        if zt < 0.:
            zt = -zt
        a = uniform(0, 1)
        while a > zt:
            theta = np.arccos(uniform(-1, 1))
            phi = np.random.uniform(0, 2*np.pi)
            xt = np.sin(theta)*np.cos(phi)
            yt = np.sin(theta)*np.sin(phi)
            zt = np.cos(theta)
            if zt < 0.:
                zt = -zt
            a = uniform(0, 1)
        x = np.append(x, xt)
        y = np.append(y, yt)
        z = np.append(z, zt)
    V = []
    for i in np.arange(n):
        v = [x[i], y[i], z[i]]
        V.append(v)
    return V


'''ReTransform (RT): given points and vectors, transforms from one coordinate system (given by thet and origin with respect to GLOBAL coordinate system) to another coordinate system (given by thet and origin with respect to GLOBAL coordinate system) '''


def RT(sourcepoints, v1, sourcethet1, ellipseorigin1, sourcethet2, ellipseorigin2):
    if len(sourcepoints) == 0:
        return [], []
    spx, spy, spz = sep(sourcepoints)
    vx, vy, vz = sep(v1)
    vectorigin = [0, 0, 0]  # don't shift vectors
    # LOCAL to GLOBAL
    vGx, vGy, vGz = transformLG(vx, vy, vz, vectorigin, sourcethet1)
    spGx, spGy, spGz = transformLG(spx, spy, spz, ellipseorigin1, sourcethet1)
    # GLOBAL back to SECOND LOCAL
    vfx, vfy, vfz = transformGL(vGx, vGy, vGz, vectorigin, sourcethet2)
    spfx, spfy, spfz = transformGL(
        spGx, spGy, spGz, ellipseorigin2, sourcethet2)
    sp = sepop(spfx, spfy, spfz)
    v2 = sepop(vfx, vfy, vfz)
    return sp, v2


'''Select Range specifically for ellipse 7 (see page 131 sheets) and 143.'''


def SR10(xrange, X, Y, Z, origin):
    xintG = []
    yintG = []
    zintG = []
    for i in range(0, len(X)):
        xinti = X[i]
        yinti = Y[i]
        zinti = Z[i]
        if (xinti-origin[0])**2 + (zinti)**2 <= xrange**2 and yinti < 0:
            xintG.append(xinti)
            yintG.append(yinti)
            zintG.append(zinti)
    return xintG, yintG, zintG


def SR7(xrange, X, Y, Z, origin):
    xintG = []
    yintG = []
    zintG = []
    for i in range(0, len(X)):
        xinti = X[i]
        yinti = Y[i]
        zinti = Z[i]
        if (xinti-origin[0])**2 + (zinti)**2 <= xrange**2 and yinti > 0:
            xintG.append(xinti)
            yintG.append(yinti)
            zintG.append(zinti)
    return xintG, yintG, zintG


'''Creates a circular source with a given radius'''


def circularsource(r, n):  # radius
    xpoint = []
    ypoint = []
    zpoint = []
    x = np.asarray([r*random.uniform(-1, 1) for a in np.random.rand(n)])
    x = x.astype(float)
    y = np.asarray([r*random.uniform(-1, 1) for a in np.random.rand(n)])
    y = y.astype(float)
    for i in range(n):
        d = (x[i]**2) + (y[i]**2)
        if d < r**2:
            xpoint.append(x[i])
            ypoint.append(y[i])
            zpoint.append(0.0)
    return xpoint, ypoint, zpoint


'''generates random points within a circle '''


def circ2(r):
    x = np.asarray([r*random.uniform(-1, 1) for a in np.random.rand(10)])
    x = x.astype(float)
    y = np.asarray([r*random.uniform(-1, 1) for a in np.random.rand(10)])
    y = y.astype(float)
    for i in range(10):
        d = (x[i]**2) + (y[i]**2)
        if d < r**2:
            return x[i], y[i]


''' A different function that generates n points within a circle with radius r'''


def circularsource1(r, n):  # radius
    xpoint = []
    ypoint = []
    zpoint = []
    for i in range(n):
        x, y = circ2(r)
        xpoint.append(x)
        ypoint.append(y)
        zpoint.append(0.0)
    return xpoint, ypoint, zpoint


'''FormSource: (fixed) give number of rays, potential source points, the GLOBAL angle, and GLOBAL origin (that the source should be made with respect to). Returns random collection of points and vectors. '''


def FS(specnum, sourcepoint, sourcethet, origin):
    originG = [0, 0, 0]
    if type(sourcepoint[0]) is int or type(sourcepoint[0]) is float or type(sourcepoint[0]) is numpy.float64:
        v1 = spec(specnum)
        vx, vy, vz = sep(v1)
        v1x, v1y, v1z = transformLG(vx, vy, vz, originG, sourcethet)
        p1x, p1y, p1z = shift(sourcepoint, origin)
        sp = [p1x, p1y, p1z]
        v2 = sepop(v1x, v1y, v1z)
    else:
        v1 = spec(specnum)
        vx, vy, vz = sep(v1)
        v1x, v1y, v1z = transformLG(vx, vy, vz, originG, sourcethet)
        v2 = sepop(v1x, v1y, v1z)
        sp = []
        for i in range(0, specnum):
            j = random.randint(0, len(sourcepoint[0])-1)
            spT = [sourcepoint[0][j], sourcepoint[1][j], sourcepoint[2][j]]
            sp.append(spT)
    return sp, v2


'''creates a list of potential source points within a certain range, tilted at a certain angle, corresponding to a specific origin.
def specsource(r,origin,thet,n):
    if r ==0.0:
        return origin
    x,y,z=circularsource(r,n)
    x1,y1,z1 = transformLG(x,y,z,origin,thet)
    sourcepoint = [x1,y1,z1]
    return sourcepoint '''

'''from the xrange being determined in GLOBAL coordinate system, translates
from Global to Local. Returns center point and the xrange'''


def xrangeGL6(x1, y1, z1, x3, y3, z3, origin, thet):
    x, y, z = [], [], []
    x2, y2, z2 = transformGL(x1, y1, z1, origin, thet)
    x4, y4, z4 = transformGL(x3, y3, z3, origin, thet)
    x.extend(x2), x.extend(x4), y.extend(
        y2), y.extend(y4), z.extend(z2), z.extend(z4)

    xrangeL = np.sqrt((min(x) - max(x))**2)/2
    yrangeL = np.sqrt((min(y) - max(y))**2)/2
    zrangeL = np.sqrt((min(z) - max(z))**2)/2

    xcenter = min(x) + xrangeL
    ycenter = min(y) + yrangeL
    zcenter = min(z) + zrangeL

    xcenterL = [xcenter, ycenter, zcenter]
    xrangesL = [xrangeL, yrangeL, zrangeL]
    return xcenterL, xrangesL


'''from the xrange being determined in GLOBAL coordinate system, translates
from Global to Local. Returns center point and the xrange. BUT random yrange to
maximize area covered. (see fixing xrangeGL6)'''


def xrangeGL7(x1, y1, z1, x3, y3, z3, origin, thet):
    x, y, z = [], [], []
    x2, y2, z2 = transformGL(x1, y1, z1, origin, thet)
    x4, y4, z4 = transformGL(x3, y3, z3, origin, thet)
    x.extend(x2), x.extend(x4), y.extend(
        y2), y.extend(y4), z.extend(z2), z.extend(z4)

    xrangeL = np.sqrt((min(x) - max(x))**2)/2
    yrangeL = np.sqrt((min(y) - max(y))**2)/2
    zrangeL = np.sqrt((min(z) - max(z))**2)/2

    xcenter = min(x) + xrangeL
    ycenter = min(y) + yrangeL
    zcenter = min(z) + zrangeL

    xcenterL = [xcenter, ycenter, zcenter]
    xrangesL = [xrangeL, 200, zrangeL]
    return xcenterL, xrangesL


def SR103di(ranges, X, Y, Z, origin):  # this corrects SR103d
    xintG = []
    yintG = []
    zintG = []
    xr = ranges[0]
    yr = ranges[1]
    zr = ranges[2]
    xc = origin[0]
    yc = origin[1]
    zc = origin[2]
    for i in range(0, len(X)):
        xinti = X[i]
        yinti = Y[i]
        zinti = Z[i]
        if (((xinti-xc)**2)/(xr**2) + ((yinti-yc)**2)/(yr**2) <= 1
            and ((yinti-yc)**2)/(yr**2) + ((zinti-zc)**2)/(zr**2) <= 1
                and ((zinti-zc)**2)/(zr**2) + ((xinti-xc)**2)/(xr**2) <= 1):
            xintG.append(xinti)
            yintG.append(yinti)
            zintG.append(zinti)
    return xintG, yintG, zintG


'''given original point and vector from it, figure out properties of intersection wanted returns x2 greater (G) than or less (L) than x1'''


def finddirec(p1, v, intpos, intneg, vectpos, vectneg):
    x1 = p1[0]
    y1 = p1[1]
    z1 = p1[2]
    v1 = v[0]
    v2 = v[1]
    v3 = v[2]
    xpos = intpos[0]
    ypos = intpos[1]
    zpos = intpos[2]
    xneg = intneg[0]
    yneg = intneg[1]
    zneg = intneg[2]
    direc = []
    direcpos = []
    direcneg = []
    if v1 >= 0:
        direc.append('G')
    if v1 < 0:
        direc.append('L')
    if v2 >= 0:
        direc.append('G')
    if v2 < 0:
        direc.append('L')
    if v3 >= 0:
        direc.append('G')
    if v3 < 0:
        direc.append('L')
    if xpos >= x1:
        direcpos.append('G')
    if xpos < x1:
        direcpos.append('L')
    if ypos >= y1:
        direcpos.append('G')
    if ypos < y1:
        direcpos.append('L')
    if zpos >= z1:
        direcpos.append('G')
    if zpos < z1:
        direcpos.append('L')
    if xneg >= x1:
        direcneg.append('G')
    if xneg < x1:
        direcneg.append('L')
    if yneg >= y1:
        direcneg.append('G')
    if yneg < y1:
        direcneg.append('L')
    if zneg >= z1:
        direcneg.append('G')
    if zneg < z1:
        direcneg.append('L')
    if direc == direcpos:
        return intpos, vectpos
    else:
        if direc == direcneg:
            return intneg, vectneg


'''Reflection of a ray off of an ellipse using ELI2. '''


def REPCNi(coeffellipse, pli, v):
    Npos, Nneg = NormalP(pli, v, coeffellipse)  # plane coefficients
    VectLNorm = N(v)  # incident unit vector
    Npos = np.array([-x for x in Npos])
    Nneg = np.array([-x for x in Nneg])
    vectpos = VectLNorm - 2*N(Npos)*(np.dot(VectLNorm, N(Npos)))
    vectneg = VectLNorm - 2*N(Nneg)*(np.dot(VectLNorm, N(Nneg)))
    xint, yint, zint = ELI2(pli, v, coeffellipse)
    intpos = [float(xint[0]), float(yint[0]), float(
        zint[0])]  # array and points of intersection
    intneg = [float(xint[1]), float(yint[1]), float(
        zint[1])]  # array and points of intersection
    GoodInt, GoodVect = finddirec(pli, v, intpos, intneg, vectpos, vectneg)
    return GoodInt, GoodVect


'''reflection of a source (mulitple rays). uses 3d ellipses for range, ellipse origin for center of DESIRED range, '''


def RSEPCNi(coeffellipse, pli, vectors, ranges, ellipseorigin):
    Vect = []
    pointints = []
    if len(pli) == 0:
        return [], []
    if type(pli[0]) is int or type(pli[0]) is float:  # assuming it is a source from one point
        for i in range(0, len(vectors)):
            Gpoint, Gvect = REPCNi(coeffellipse, pli, vectors[i])
            if SR3B(ranges, Gpoint[0], Gpoint[1], Gpoint[2], ellipseorigin) == True:
                pointints.append(Gpoint)
                Vect.append(Gvect)
    else:
        for i in range(0, len(pli)):
            Vi = vectors[i]
            Pli = pli[i]  # (or pli/original points of lines)
            Gpoint, Gvect = REPCNi(coeffellipse, Pli, Vi)
            if SR3B(ranges, Gpoint[0], Gpoint[1], Gpoint[2], ellipseorigin) == True:
                pointints.append(Gpoint)
                Vect.append(Gvect)
    return pointints, Vect


''' PlaneLineIntersectionz(PLINT): given a plane z = a number, finds intersection points of all rays'''


def PLINTz(z, p, v):
    points = []
    for i in range(0, len(p)):
        t = (z - p[i][2])/v[i][2]
        xi = p[i][0] + t*v[i][0]
        yi = p[i][1] + t*v[i][1]
        points.append([xi, yi, z])
    return points


''' PlaneLineIntersection(PLINT): given a plane y = a number, finds intersection points of all rays'''


def PLINTy(y, p, v):
    points = []
    for i in range(0, len(p)):
        t = (y - p[i][1])/v[i][1]
        xi = p[i][0] + t*v[i][0]
        zi = p[i][2] + t*v[i][2]
        points.append([xi, y, zi])
    return points


'''select region mirror. if a point is within the ellipse of a mirror, return true.'''


def SRM(p, coeffmirr, origin):
    X = p[0]
    Z = p[2]
    if ((((X-origin[0])**2)/coeffmirr[1]**2) + ((Z-origin[2])**2)/coeffmirr[0]**2) <= 1:
        return True
    return False


'''find intersection points of rays and the mirror.'''


def IntM(p, v, coeffmirr, originmirr):
    hitints = []
    hitvects = []
    missints = []
    missvects = []
    intpoints = PLINTy(originmirr[1], p, v)
    for i in range(0, len(intpoints)):
        if SRM(intpoints[i], coeffmirr, originmirr) == True:
            hitints.append(intpoints[i])
            VectLNorm = N(v[i])
            PNorm = [0, -1, 0]  # from definition of mirror (check sign what)
            VectReflect = VectLNorm - 2*N(PNorm)*(np.dot(VectLNorm, N(PNorm)))
            hitvects.append(VectReflect)  # change to reflected
        else:
            missints.append(intpoints[i])
            missvects.append(v[i])
    return hitints, hitvects, missints, missvects


''' for ONE ray'''


def PLINTyS(y, p, v):
    t = (y - p[1])/v[1]
    xi = p[0] + t*v[0]
    zi = p[2] + t*v[2]
    return(xi, y, zi)


''' for ONE ray'''


def PLINTzS(z, p, v):
    t = (z - p[i][2])/v[i][2]
    xi = p[i][0] + t*v[i][0]
    yi = p[i][1] + t*v[i][1]
    return [xi, yi, z]


'''find intersection points of one ray and the mirror.'''


def IntMS(p, v, coeffmirr, originmirr):
    '''test is a test'''
    hitints = []
    hitvects = []
    missints = []
    missvects = []
    intpoint = PLINTyS(originmirr[1], p, v)
    if SRM(intpoint, coeffmirr, originmirr) == True:
        hitints = intpoint
        VectLNorm = N(v)
        PNorm = [0, -1, 0]  # from definition of mirror (check sign what)
        VectReflect = VectLNorm - 2*N(PNorm)*(np.dot(VectLNorm, N(PNorm)))
        hitvects = VectReflect  # change to reflected
    else:
        missints = intpoint
        missvects = v
    return hitints, hitvects, missints, missvects


'''plotting the mirror '''


def mirror(origin, coeffmirr, y):
    px = []
    pz = []
    py = []
    X = np.linspace(-coeffmirr[1], coeffmirr[1], 50)
    Z = np.linspace(-coeffmirr[0], coeffmirr[0], 50)
    for i in range(50):
        x = X[i]
        for j in range(50):
            z = Z[j]
            if ((((x-origin[0])**2)/coeffmirr[1]**2) + ((z-origin[2])**2)/coeffmirr[0]**2) < 1:
                px.append(x)
                pz.append(z)
                py.append(y)
    return px, py, pz


'''plotting the polarizers '''


def polarizer(origin, coeffmirr, y):
    px = []
    pz = []
    py = []
    X = np.linspace(-coeffmirr[1], coeffmirr[1], 50)
    Z = np.linspace(-coeffmirr[0], coeffmirr[0], 50)
    for i in range(50):
        x = X[i]
        for j in range(50):
            z = Z[j]
            if ((((x)**2)/coeffmirr[1]**2) + ((z)**2)/coeffmirr[0]**2) < 1:
                px.append(x)
                pz.append(z)
                py.append(y)
    thet = [0, 0, 0]
    px, py, pz = transformLG(px, py, pz, origin, thet)
    return px, py, pz


''' EllipseLineInt(ELI): Give point of the line, vector of the line, and coefficients of the ellipse, find the intersection(s) of the line and the ellipsoid (assuming ellipse is rotated about the x-axis.
This is where (x-x_0)/a = (y-y_0)/b) = (z-z_0)/c. And, x^2/d^2 + y^2/e^2 + z^2/e^2 = 1 (ellipse is rotated around x axis to form ellipsoid. Different version of solving equation. '''


def ELI3(pli, v1, coeffellipse):
    x0, y0, z0, vx, vy, vz, a, b = ELIorganize(pli, v1, coeffellipse)
    A = 1/(a**2) + ((vy**2)/((b**2)*(vx**2))) + ((vz**2)/((b**2)*(vx**2)))
    B1 = ((-2*x0*(vy**2))/((b**2)*(vx**2))) + ((2*vy*y0)/((b**2)*(vx)))
    B2 = ((-2*x0*(vz**2))/((b**2)*(vx**2))) + ((2*vz*z0)/((b**2)*(vx)))
    B = B1 + B2
    C1 = (((x0**2)*(vy**2))/((b**2)*(vx**2))) + \
        ((-2*x0*vy*y0)/((b**2)*(vx))) + ((y0**2)/(b**2))
    C2 = (((x0**2)*(vz**2))/((b**2)*(vx**2))) + \
        ((-2*x0*vz*z0)/((b**2)*(vx))) + ((z0**2)/(b**2))
    C = C1 + C2 - 1.0
    xint = [(-B + np.sqrt((B**2)-4*A*C))/(2*A),
            (-B - np.sqrt((B**2)-4*A*C))/(2*A)]
    t = [(xint[0]-x0)/vx, (xint[1]-x0)/vx]
    yint = [y0 + t[0]*vy, y0 + t[1]*vy]
    zint = [z0 + t[0]*vz, z0 + t[1]*vz]
    return xint, yint, zint


''' reflection of a ray off of an ellipse using ELI3 '''


def REPCNi3(coeffellipse, pli, v):
    Npos, Nneg = NormalP(pli, v, coeffellipse)  # plane coefficients
    VectLNorm = N(v)  # incident unit vector
    Npos = np.array([-x for x in Npos])
    Nneg = np.array([-x for x in Nneg])
    vectpos = VectLNorm - 2*N(Npos)*(np.dot(VectLNorm, N(Npos)))
    vectneg = VectLNorm - 2*N(Nneg)*(np.dot(VectLNorm, N(Nneg)))
    xint, yint, zint = ELI3(pli, v, coeffellipse)
    intpos = [float(xint[0]), float(yint[0]), float(
        zint[0])]  # array and points of intersection
    intneg = [float(xint[1]), float(yint[1]), float(
        zint[1])]  # array and points of intersection
    GoodInt, GoodVect = finddirec(pli, v, intpos, intneg, vectpos, vectneg)
    return GoodInt, GoodVect


''' Just gives randomized polarization'''


def InitialPolarization():
    A = random.random()
    thet = A*2*np.pi
    Eox = np.cos(thet)
    Eoy = np.sin(thet)
    return Eox, Eoy, thet  # this is AMPLITUDE OF COMPONENTS, NOT INTENSITY.


''' for ONE ray'''


def PLINTyS(y, p, v):
    t = (y - p[1])/v[1]
    xi = p[0] + t*v[0]
    zi = p[2] + t*v[2]
    return(xi, y, zi)


'''given two angles (of polarization and polarizer) returns the intensity of reflected'''


def PolarizerInteractionR(Eox, Eoy, thet_polarized, PolarizerAngle):
    I = Eox**2 + Eoy**2  # here I IS intensity
    thet_p = PolarizerAngle
    thet_alpha = thet_polarized - thet_p
    return I*np.cos(thet_alpha)**2  # Intensity REFLECTED.


'''given two angles (of polarization and polarizer) returns the intensity of transmitted'''


def PolarizerInteractionT(Eox, Eoy, thet_polarized, PolarizerAngle):
    I = Eox**2 + Eoy**2  # INTENSITY
    thet_p = PolarizerAngle
    thet_alpha = thet_polarized - thet_p
    return I*np.sin(thet_alpha)**2  # INTENSITY TRANSMITTED


'''returns distance between two given points'''


def dist(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)


'''Given initial ray and polarizer, return the TRANSMITTED RAY. All in Global Coordinates.
If the ray ever misses the next stop, it returns 0 and is discarded '''


# @jit(cache=True)
def IntPolT2(Ray, coeffpolar, originpolar, PolarizerAngle):  # transmitted
    if Ray is None:
        return
    thet_polarized = Ray[0]  # theta
    I = Ray[1]  # intensity
    p = Ray[2]  # point
    v = Ray[3]  # vector
    Di = Ray[4]  # distance
    Ex, Ey = np.sqrt(I)*np.cos(thet_polarized), np.sqrt(I) * \
        np.sin(thet_polarized)  # now AMPLITUDE
    Ray_T = []  # thet, I, intpoint, vects
    intpoint = PLINTyS(originpolar[1], p, v)
    Ray_T.append(PolarizerAngle+np.pi/2)
    I_T = PolarizerInteractionT(Ex, Ey, thet_polarized, PolarizerAngle)
    Ray_T.append(I_T)
    if SRM(intpoint, coeffpolar, originpolar) == True:
        Ray_T.append(intpoint)
        Ray_T.append(v)  # just transmitted as same vector (assuming)
        Df = dist(p, intpoint)
        Ray_T.append(Di + Df)
    else:
        return
    return Ray_T


'''Given initial ray and polarizer, return the REFLECTED RAY. All in Global Coordinates.
If the ray ever misses the next stop, it returns 0 and is discarded '''


# @jit(cache=True)
def IntPolR2(Ray, coeffpolar, originpolar, PolarizerAngle):  # reflected
    if Ray is None:  # just bug check
        return
    thet_polarized = Ray[0]  # theta
    I = Ray[1]  # intensity
    p = Ray[2]  # point
    v = Ray[3]  # vector
    Di = Ray[4]  # distance
    Ex, Ey = np.sqrt(I)*np.cos(thet_polarized), np.sqrt(I) * \
        np.sin(thet_polarized)
    Ray_R = []  # thet, I, intpoint, vects
    intpoint = PLINTyS(originpolar[1], p, v)
    Ray_R.append(PolarizerAngle)  # same vector of course
    I_R = PolarizerInteractionR(Ex, Ey, thet_polarized, PolarizerAngle)
    Ray_R.append(I_R)
    if SRM(intpoint, coeffpolar, originpolar) == True:
        Ray_R.append(intpoint)
        VectLNorm = N(v)
        PNorm = [0, -1, 0]  # from definition of mirror (check sign what)
        VectReflect = VectLNorm - 2*N(PNorm)*(np.dot(VectLNorm, N(PNorm)))
        Ray_R.append(VectReflect)  # change to reflected
        Df = dist(p, intpoint)
        Ray_R.append(Di + Df)
    else:
        return
    return Ray_R


'''find intersection points of given ray and the mirror. (ignoring missing rays)'''


# @jit(cache=True)
def IntM2(Ray, coeffmirr, originmirr):
    if Ray is None:
        return
    p = Ray[2]
    v = Ray[3]
    Ray_M = []
    Ray_M.append(Ray[0] + np.pi)  # flips TO BE CONSISTENT
    Ray_M.append(Ray[1])
    intpoint = PLINTyS(originmirr[1], p, v)
    if SRM(intpoint, coeffmirr, originmirr) == True:
        Ray_M.append(intpoint)
        VectLNorm = N(v)
        PNorm = [0, -1, 0]  # from definition of mirror (check sign what)
        VectReflect = VectLNorm - 2*N(PNorm)*(np.dot(VectLNorm, N(PNorm)))
        Ray_M.append(VectReflect)  # change to reflected
        Df = dist(p, intpoint)  # ADDING DISTANCE TRAVELLED (ADDED 4/21)
        Ray_M.append(Ray[4]+Df)
        return Ray_M
    else:
        return


'''gives a ray (polarization, intensity = 1, point, vector, distnace) with a random angle from a point.'''


def CreateRay():
    Ex, Ey, thet1 = InitialPolarization()  # picks arbitrary thet and intensity 1
    sourcepoint = [-160.375, -113, 0]  # global
    # angle (global)
    rand = float(random.randrange(32000, 96000))
    angle = rand/1000
    v = [angle, 251, 0]  # random angle
    Ray = [thet1, 1.0, sourcepoint, v, 0]
    return Ray


'''extending create ray into z plane'''


def CreateRay3D():
    Ex, Ey, thet1 = InitialPolarization()  # picks arbitrary thet and intensity 1
    sourcepoint = [-160.375, -113, 0]  # global
    rand = float(random.randrange(32000, 96000))
    angle = rand/1000
    rand2 = float(random.randrange(32000, 96000))
    angle2 = rand2/2000
    v = [angle, 251, angle2]  # random angle
    Ray = [thet1, 1.0, sourcepoint, v, 0]
    return Ray


'''Give ray and everything in global, does work in local REFLECTING OFF OF
GIVEN COEFFELLIPSE, returns in global: ORIGINAL NOT INCLUDING RANGE'''


def ReflEllO(Ray, thetL, originL, coeffellipse):
    Ray_Refl = []
    originG = [0, 0, 0]  # the global origin
    thetG = [0, 0, 0]  # rotation with respect to itself aka 0,0,0
    sourcepoint = Ray[2]  # originalpoint
    v = Ray[3]  # vector
    # point and vector in local coordinates
    SPLi, VPLi = RT(sourcepoint, v, thetG, originG, thetL, originL)
    pointsf, vectsf = REPCNi3(coeffellipse, SPLi, VPLi)
    SPLf, VPLf = RT(pointsf, vectsf, thetL, originL, thetG, originG)
    # + np.pi) maybe no change in polarization when reflecting?/not flipped?
    Ray_Refl.append(Ray[0])
    Ray_Refl.append(Ray[1])
    Ray_Refl.append(SPLf)
    Ray_Refl.append(VPLf)
    Df = dist(SPLi, SPLf)
    Ray_Refl.append(Ray[4] + Df)
    return Ray_Refl


'''REFLECTION OFF ELLIPSOIDS WITH RANGE INCLUDED '''


def ReflEll(Ray, thetL, originL, coeffellipse, center, ranges):
    if Ray is None:
        return
    Ray_Refl = []
    originG = [0, 0, 0]  # the global origin
    thetG = [0, 0, 0]  # rotation with respect to itself aka 0,0,0
    sourcepoint = Ray[2]  # originalpoint
    v = Ray[3]  # vector
    # point and vector in local coordinates
    SPLi, VPLi = RT(sourcepoint, v, thetG, originG, thetL, originL)
    # IN LOCAL COORDINATES, REFLECTION OFF ELLIPSOID
    pointsf, vectsf = REPCNi(coeffellipse, SPLi, VPLi)
    if SR3B(ranges, pointsf[0], pointsf[1], pointsf[2], center) == True:
        SPLf, VPLf = RT(pointsf, vectsf, thetL, originL, thetG,
                        originG)  # IN LOCAL COORDINATES,
        Ray_Refl.append(Ray[0] + np.pi)
        Ray_Refl.append(Ray[1])
        Ray_Refl.append(SPLf)
        Ray_Refl.append(VPLf)
        # SO THE DISTANCE BETWEEN ORIGINAL POINT AND INTERSECTION OF ELLIPSOID.
        Df = dist(sourcepoint, SPLf)
        Ray_Refl.append(Ray[4] + Df)  # should fuckin work.
        return Ray_Refl
    else:
        return


'''given a vector, checks the angle'''


def checkangle(v):
    x, y, z = v[0], v[1], v[2]
    theta = np.arctan(np.sqrt((x**2)+(y**2))/z)
    if theta <= .463647609:
        return True
    else:
        return False


'''Gives one restricted specular angle (restricted in that it will hit) '''


def ORS():  # one restricted spec
    x, y, z = [], [], []
    theta = np.arccos(uniform(-1, 1))
    phi = np.random.uniform(0, 2*np.pi)
    xt = np.sin(theta)*np.cos(phi)
    yt = np.sin(theta)*np.sin(phi)
    zt = np.cos(theta)
    if zt < 0.:
        zt = -zt
    a = uniform(0, 1)
    while a > zt:
        theta = np.arccos(uniform(-1, 1))
        phi = np.random.uniform(0, 2*np.pi)
        xt = np.sin(theta)*np.cos(phi)
        yt = np.sin(theta)*np.sin(phi)
        zt = np.cos(theta)
        if zt < 0.:
            zt = -zt
        a = uniform(0, 1)
    v = [xt, yt, zt]
    while checkangle(v) == True:
        return v
    if checkangle(v) == False:
        return ORS()


'''given a vector, checks the angle to make < lim degrees'''


def checkangle_narrow(v, lim):
    # lim = np.pi/6 #30 degrees
    # lim= np.pi/9 #20 degrees
    # lim = np.pi/18 #10 degrees
    x, y, z = v[0], v[1], v[2]
    theta = np.pi/2 - np.arctan(z/np.sqrt((x**2)+(y**2)))
    if np.abs(theta) <= lim:
        return True
    else:
        return False


'''Gives one restricted specular angle (restricted in that it will hit AND is
within 30 degrees of the z axis) '''


def ORS_narrow(lim):  # one restricted spec
    x, y, z = [], [], []
    theta = np.arccos(uniform(-1, 1))
    phi = np.random.uniform(0, 2*np.pi)
    xt = np.sin(theta)*np.cos(phi)
    yt = np.sin(theta)*np.sin(phi)
    zt = np.cos(theta)
    if zt < 0.:
        zt = -zt
    a = uniform(0, 1)
    while a > zt:
        theta = np.arccos(uniform(-1, 1))
        phi = np.random.uniform(0, 2*np.pi)
        xt = np.sin(theta)*np.cos(phi)
        yt = np.sin(theta)*np.sin(phi)
        zt = np.cos(theta)
        if zt < 0.:
            zt = -zt
        a = uniform(0, 1)
    v = [xt, yt, zt]
    while checkangle_narrow(v, lim) == True:
        return v
    if checkangle_narrow(v, lim) == False:
        return ORS_narrow(lim)


'''returns a certain number (n) of random distribution of restricted specular rays '''


def specRestricted(n):
    V = []
    for i in np.arange(n):
        v = ORS()
        V.append(v)
    return V


'''returns a certain number (n) of random distribution of restricted specular
rays < 30 degrees'''


def specRestricted_narrow(n, lim):
    V = []
    for i in np.arange(n):
        v = ORS_narrow(lim)
        V.append(v)
    return V


'''creates a specular source within a range, made of points and vectors '''


def FS(specnum, sourcepoint, sourcethet, origin):
    originG = [0, 0, 0]
    if type(sourcepoint[0]) is int or type(sourcepoint[0]) is float or type(sourcepoint[0]) is numpy.float64:
        v1 = spec(specnum)
        vx, vy, vz = sep(v1)
        v1x, v1y, v1z = transformLG(vx, vy, vz, originG, sourcethet)
        p1x, p1y, p1z = shift(sourcepoint, origin)
        sp = [p1x, p1y, p1z]
        v2 = sepop(v1x, v1y, v1z)
    else:
        v1 = spec(specnum)
        vx, vy, vz = sep(v1)
        v1x, v1y, v1z = transformLG(vx, vy, vz, originG, sourcethet)
        v2 = sepop(v1x, v1y, v1z)
        sp = []
        for i in range(0, specnum):
            j = random.randint(0, len(sourcepoint[0])-1)
            spT = [sourcepoint[0][j], sourcepoint[1][j], sourcepoint[2][j]]
            sp.append(spT)
    return sp, v2


'''Creates an INITIAL ray with polarization, intensity, point, vector, and distance travelled '''


def CreateRay3D():
    Ex, Ey, thet1 = InitialPolarization()  # picks arbitrary thet and intensity 1
    sourcepoint = [-160.375, -113, 0]  # global
    rand = float(random.randrange(32000, 96000))
    angle = rand/1000
    rand2 = float(random.randrange(32000, 96000))
    angle2 = rand2/2000
    v = [angle, 251, angle2]  # random angle
    Ray = [thet1, 1.0, sourcepoint, v, 0]
    return Ray


'''creates a SOURCE OF INITIAL RAYS, with rays defined by point and vector,
arbitrary polarization, and distance travelled 0'''


def FSRay(specnum, sourcepoint, sourcethet, origin):
    originG = [0, 0, 0]
    Rays = []
    if type(sourcepoint[0]) is int or type(sourcepoint[0]) is float or type(sourcepoint[0]) is numpy.float64:
        for i in range(0, specnum):
            v1 = specRestricted(1)
            vx, vy, vz = sep(v1)
            v1x, v1y, v1z = transformLG(vx, vy, vz, originG, sourcethet)
            v2 = sepop(v1x, v1y, v1z)
            Ex, Ey, thet1 = InitialPolarization()
            spT = [sourcepoint[0], sourcepoint[1], sourcepoint[2]]
            Ray = [thet1, 1.0, spT, v2[0], 0]
            Rays.append(Ray)
        return Rays
    else:
        for i in range(0, specnum):
            v1 = specRestricted(1)
            vx, vy, vz = sep(v1)
            v1x, v1y, v1z = transformLG(vx, vy, vz, originG, sourcethet)
            v2 = sepop(v1x, v1y, v1z)
            j = random.randint(0, len(sourcepoint[0])-1)
            spT = [sourcepoint[0][j], sourcepoint[1][j], sourcepoint[2][j]]
            Ex, Ey, thet1 = InitialPolarization()
            Ray = [thet1, 1.0, spT, v2[0], 0]
            Rays.append(Ray)
    return Rays


'''creates a SOURCE OF INITIAL RAYS, with rays defined by point and vector,
arbitrary polarization, and distance travelled 0, < 30 degree launch angle'''


def FSRay_narrow(specnum, sourcepoint, sourcethet, origin, lim):
    originG = [0, 0, 0]
    Rays = []
    if type(sourcepoint[0]) is int or type(sourcepoint[0]) is float or type(sourcepoint[0]) is numpy.float64:
        for i in range(0, specnum):
            v1 = specRestricted_narrow(1, lim)
            vx, vy, vz = sep(v1)
            v1x, v1y, v1z = transformLG(vx, vy, vz, originG, sourcethet)
            v2 = sepop(v1x, v1y, v1z)
            Ex, Ey, thet1 = InitialPolarization()
            spT = [sourcepoint[0], sourcepoint[1], sourcepoint[2]]
            Ray = [thet1, 1.0, spT, v2[0], 0]
            Rays.append(Ray)
        return Rays
    else:
        for i in range(0, specnum):
            v1 = specRestricted_narrow(1, lim)
            vx, vy, vz = sep(v1)
            v1x, v1y, v1z = transformLG(vx, vy, vz, originG, sourcethet)
            v2 = sepop(v1x, v1y, v1z)
            j = random.randint(0, len(sourcepoint[0])-1)
            spT = [sourcepoint[0][j], sourcepoint[1][j], sourcepoint[2][j]]
            Ex, Ey, thet1 = InitialPolarization()
            Ray = [thet1, 1.0, spT, v2[0], 0]
            Rays.append(Ray)
    return Rays


'''SAME AS ABOVE BUT INITIAL INITIAL POLARIZATION VALUE ZERO'''


def FSRay_Zero(specnum, sourcepoint, sourcethet, origin):
    originG = [0, 0, 0]
    Rays = []
    if type(sourcepoint[0]) is int or type(sourcepoint[0]) is float or type(sourcepoint[0]) is numpy.float64:
        for i in range(0, specnum):
            v1 = specRestricted(1)
            vx, vy, vz = sep(v1)
            v1x, v1y, v1z = transformLG(vx, vy, vz, originG, sourcethet)
            v2 = sepop(v1x, v1y, v1z)
            # Ex,Ey,thet1 = InitialPolarization()
            thet1 = 0
            spT = [sourcepoint[0], sourcepoint[1], sourcepoint[2]]
            Ray = [thet1, 1.0, spT, v2[0], 0]
            Rays.append(Ray)
        return Rays
    else:
        for i in range(0, specnum):
            v1 = specRestricted(1)
            vx, vy, vz = sep(v1)
            v1x, v1y, v1z = transformLG(vx, vy, vz, originG, sourcethet)
            v2 = sepop(v1x, v1y, v1z)
            j = random.randint(0, len(sourcepoint[0])-1)
            spT = [sourcepoint[0][j], sourcepoint[1][j], sourcepoint[2][j]]
            # Ex,Ey,thet1 = InitialPolarization()
            thet1 = 0
            Ray = [thet1, 1.0, spT, v2[0], 0]
            Rays.append
        return Rays


'''SAME AS ABOVE BUT INITIAL INITIAL POLARIZATION VALUE ZERO < 30 degree launch angle'''


def FSRay_Zero_narrow(specnum, sourcepoint, sourcethet, origin, lim):
    originG = [0, 0, 0]
    Rays = []
    if type(sourcepoint[0]) is int or type(sourcepoint[0]) is float or type(sourcepoint[0]) is numpy.float64:
        for i in range(0, specnum):
            v1 = specRestricted_narrow(1, lim)
            vx, vy, vz = sep(v1)
            v1x, v1y, v1z = transformLG(vx, vy, vz, originG, sourcethet)
            v2 = sepop(v1x, v1y, v1z)
            # Ex,Ey,thet1 = InitialPolarization()
            thet1 = 0
            spT = [sourcepoint[0], sourcepoint[1], sourcepoint[2]]
            Ray = [thet1, 1.0, spT, v2[0], 0]
            Rays.append(Ray)
        return Rays
    else:
        for i in range(0, specnum):
            v1 = specRestricted_narrow(1, lim)
            vx, vy, vz = sep(v1)
            v1x, v1y, v1z = transformLG(vx, vy, vz, originG, sourcethet)
            v2 = sepop(v1x, v1y, v1z)
            j = random.randint(0, len(sourcepoint[0])-1)
            spT = [sourcepoint[0][j], sourcepoint[1][j], sourcepoint[2][j]]
            # Ex,Ey,thet1 = InitialPolarization()
            thet1 = 0
            Ray = [thet1, 1.0, spT, v2[0], 0]
            Rays.append
        return Rays


'''creates a circle of points in the x-y plane '''


def circle(c, r):
    x = np.linspace((c[0]-r+.000000001), (c[0]+r-.000000001), 50)
    yp = []
    yn = []
    for i in range(50):
        y1 = np.sqrt((r**2)-(x[i]-c[0])**2)+c[1]
        y2 = -np.sqrt((r**2)-(x[i]-c[0])**2)+c[1]
        yp.append(y1)
        yn.append(y2)
    return x, yp, yn


'''creates n source points within a certain range, at a tilt angle of a given
origin in global coordinates'''


def specsource(r, origin, thet, n):
    if r == 0.0:
        return origin
    x, y, z = circularsource(r, n)
    x1, y1, z1 = transformLG(x, y, z, origin, thet)
    sourcepoint = [x1, y1, z1]
    return sourcepoint


'''returns point of intersection of a vector with the xy plane located at z'''


def PLINTzS(z, p, v):
    t = (z - p[2])/v[2]
    xi = p[0] + t*v[0]
    yi = p[1] + t*v[1]
    point = [xi, yi, z]
    return point


def makefocusedvect(sourcepoint, point):
    dx = point[0]-sourcepoint[0]
    dy = point[1]-sourcepoint[1]
    dz = point[2]-sourcepoint[2]
    vFocus = N([dx, dy, dz])
    return vFocus


# THESE are functions wihout movement of the mirror
'''def fillcircle(r,center,num):
    ppr = int((4/np.pi)*np.sqrt(num)
              ) #number of sections needed per quarter circle#points (sections) per radius
    xstart = (center[0] - r)
    xend = (center[0] + r)
    xs = np.linspace(xstart,xend,ppr)
    ystart = (center[1]-r)
    yend = (center[1]+r)
    ys = np.linspace(ystart,yend,ppr)
    points = []
    for i in range(len(xs)):
        for j in range (len(ys)):
            if (xs[i]-center[0])**2 + (ys[j]-center[1])**2 < r**2:
                p = [xs[i],ys[j]]
                points.append(p)
    return points

def checkoutrays(Rays,center,r):
    GRays = []
    BRays = []
    for i in range(len(Rays)):
        det = PLINTzS(80.,Rays[i][2],Rays[i][3])
        Rays[i][2] = det
        d = ((det[0]-center[0])**2) + ((det[1]-center[1])**2)
        if d <= r**2:
            GRays.append(Rays[i])
        else:
            BRays.append(Rays[i])
    return GRays,BRays

'''


def gridlines(r, center, num):
    # number of sections needed per quarter circle#points (sections) per radius
    ppr = int((4/np.pi)*np.sqrt(num))
    xstart = (center[0] - r)
    xend = (center[0] + r)
    xs = np.linspace(xstart, xend, ppr)
    ystart = (center[1]-r)
    yend = (center[1]+r)
    ys = np.linspace(ystart, yend, ppr)
    return xs, ys


'''
def OFD(Rays): #output from detector
    Rayf = []
    for i in range(len(Rays)):
        Paths = [TTTTio,RRRRio,TTRRio,RTTRio,RTRTio,TRRTio,RRTTio,TRTRio]
        Ri = Rays[i]
        for j in range(8):
            out = Paths[j](Ri,p1,p2,p3,p4)
            if out is not None:
                Rayf.append(out)
    return Rayf
'''

'''Used to sort rays in pixels on the detector '''


def sortgrid(Gtest):  # assuming detector but can change
    jx, jy = gridlines(7.9375, [160.375, -113], 200)
    for i in range(len(Gtest)):
        if len(Gtest[i]) < 7:
            Gtest[i].append(0.)
            Gtest[i].append(0.)
        for j in range(len(jx)-1):
            if Gtest[i][2][0] >= jx[j] and Gtest[i][2][0] < jx[j+1]:
                Gtest[i][5] = j
            if Gtest[i][2][0] <= jx[1]:
                Gtest[i][5] = 0  # first j region
            if Gtest[i][2][0] > jx[len(jx)-2]:
                Gtest[i][5] = len(jx)-2  # last j region
        for k in range(len(jy)-1):
            if Gtest[i][2][1] >= jy[k] and Gtest[i][2][1] < jy[k+1]:
                Gtest[i][6] = k
            if Gtest[i][2][1] <= jy[1]:
                Gtest[i][6] = 0  # first j region
            if Gtest[i][2][1] > jy[len(jy)-2]:
                Gtest[i][6] = len(jy)-2  # last j region
    return Gtest


'''determines regions on the pixelated detector '''


def regionalize(Gtestsorted):
    FullRegions = []
    jx, jy = gridlines(7.9375, [160.375, -113], 200)
    for j in range(len(jx)-1):
        for k in range(len(jy)-1):
            JK = [j, k]
            for i in range(len(Gtestsorted)):
                if Gtestsorted[i][5] == j and Gtestsorted[i][6] == k:
                    JK.append(i)
            if len(JK) > 2:
                FullRegions.append(JK)
    return FullRegions


'''Runs n rays through simulation and returns rays that hit detector and the pixel region they hit  '''


def RunAll(n):  # just give number of rays to be run through this FTS
    sourcepointorigin = [-160.375, -113., -80.0]  # LOCAL
    sourcethet = [0., 0., 0.]  # SHOT STRAIGHT UP
    sourcepoints = specsource(
        7.9375, sourcepointorigin, sourcethet, n)  # LOCAL
    Rays = FSRay(n, sourcepoints, sourcethet, origin10)
    Rayf = OFD(Rays)
    G, B = checkoutrays(Rayf, [160.375, -113], 7.9375)
    Gtestsorted = sortgrid(G)
    Regions = regionalize(Gtestsorted)
    return Gtestsorted, Regions


'''returns gaussian function '''


def gaussian3d(x, y, sig, mux, muy):  # assuming is symmetric
    # sig = .3 #just guessing
    return (1/((sig**3)*(2*np.pi)**(3/2)))*np.exp(-(((x-mux)**2)/(2*sig**2) + ((y-muy)**2)/(2*sig**2)))


'''Returns power from j regions and n initial rays '''


def jRegions(n):
    OutRays, regions = RunAll(n)
    Regions = list(regions)
    jx, jy = gridlines(7.9375, [160.375, -113], 200)
    DetTot = []
    for j in range(len(regions)):
        for i in range(len(Regions[j])):  # All rays in region j
            ExTot = []
            EyTot = []
            if i != 0 and i != 1:
                JRegion = Regions[j]
                m, n = JRegion[0], JRegion[1]
                Raym = OutRays[JRegion[i]]  # ith ray in region j
                if m != len(jx)-1 and n != len(jy)-1:
                    w = gaussian3d(Raym[2][0], Raym[2][1], .4,
                                   (jx[m]+jx[m+1])/2, (jy[n]+jy[n+1])/2)
                else:
                    w = 0  # (skipping gaussian)
                Ex = np.abs(np.cos(Raym[0])*Raym[1])  # *w
                Ey = np.abs(np.sin(Raym[0])*Raym[1])  # *w
                ExTot.append(Ex)
                EyTot.append(Ey)
        Ij = (np.sum(ExTot))**2 + (np.sum(EyTot))**2
        DetTot.append(Ij)
    return DetTot

# these are the functions above but editted to include position of mirror
# from BackgroundValues import *
# from PossiblePaths import *


'''Output From Detector w/ Mirror. GIVE initial RAYS AND Y position, returns
output rays from detector if mirror at Y'''

# Each ray has 8 possible paths, yielding a possible total of 8n possible
# members of the output rays ('n' is the initial number of rays).


# We only have to call the first half of each function once since it is
# independent of the y positions. Should decrease time by 2x.

# Try using numba on the rest.
def OFDM(Rays, y):
    Rayf = []
    for i in range(len(Rays)):
        Paths = [TTTTioM, RRRRioM, TTRRioM, RTTRioM,
                 RTRTioM, TRRTioM, RRTTioM, TRTRioM]
        Ri = Rays[i]
        for j in range(8):
            origin = (0, y, 0)
            out = Paths[j](Ri, p1, p2, p3, p4, origin)
            if out is not None:
                Rayf.append(out)
    return Rayf


def ofdm_first_half(rays):
    intermediate_rays = []
    paths1 = [TTTTioM_first_half, RRRRioM_first_half, TTRRioM_first_half,
              RTTRioM_first_half, RTRTioM_first_half, TRRTioM_first_half,
              RRTTioM_first_half, TRTRioM_first_half]

    paths2 = [TTTTioM_second_half, RRRRioM_second_half, TTRRioM_second_half,
              RTTRioM_second_half, RTRTioM_second_half, TRRTioM_second_half,
              RRTTioM_second_half, TRTRioM_second_half]

    # Paths3 = [TTTTioM, RRRRioM, TTRRioM, RTTRioM,
    #           RTRTioM, TRRTioM, RRTTioM, TRTRioM]
    for ray in rays:
        for i in range(len(paths1)):
            out = paths1[i](ray, p1, p2, p3, p4)
            if out is not None:
                intermediate_rays.append({"ray": out, "path": paths2[i]})

    return intermediate_rays


def ofdm_second_half(intermediate_rays, y):
    final_rays = []
    origin = (0, y, 0)
    for ray_and_path in intermediate_rays:
        ray = ray_and_path["ray"]
        path = ray_and_path["path"]
        out = path(ray, p1, p2, p3, p4, origin)
        if out is not None:
            final_rays.append(out)

    return final_rays


'''checks if rays hit the detector or not (discards those that dont). INCLUDES
TRAVEL TO DETECTOR, returns new distance travelled, new "launch point" which is
final point on detector'''


def checkoutraysM(Rays, center, r):  # RAYS THAT HIT DETECTOR
    GRays = []
    for i in range(len(Rays)):
        det = PLINTzS(80., Rays[i][2], Rays[i][3])  # 80
        Rays[i][4] = Rays[i][4] + dist(Rays[i][2], det)
        Rays[i][2] = det
        # REACHING DETECTOR DOES NOT CHANGE POLARIZATION.
        Rays[i][0] = Rays[i][0]
        d = ((det[0]-center[0])**2) + \
            ((det[1]-center[1])**2)  # if it is within detector
        if d <= r**2:
            GRays.append(Rays[i])
    return GRays


def get_final_rays(rays, detector_center, detector_range):
    final_rays = []
    for ray in rays:
        detector_location = PLINTzS(detector_center[2], ray[2], ray[3])
        distance_from_center = dist(detector_location, detector_center)
        if (distance_from_center <= detector_range):
            new_ray = [ray[0], ray[1], ray[2], ray[3], ray[4]]
            # try keeping just the old distance!
            # print(dist(ray[2], detector_location))
            # print(ray[4])
            # print(ray[4] + dist(ray[2], detector_location))

            new_ray[4] = ray[4] + dist(ray[2], detector_location)
            new_ray[2] = detector_location
            final_rays.append(new_ray)

    return final_rays


def get_final_rays_new(rays, detector_center, detector_range):
    final_rays = []
    for ray in rays:
        detector_location = PLINTyS(detector_center[1], ray[2], ray[3])
        distance_from_center = dist(detector_location, detector_center)
        if (distance_from_center <= detector_range):
            new_ray = [ray[0], ray[1], ray[2], ray[3], ray[4]]
            new_ray[4] = ray[4] + dist(ray[2], detector_location)
            new_ray[2] = detector_location
            final_rays.append(new_ray)

    return final_rays


def get_final_rays_tilt(rays, detector_center, detector_range, normal_vec):
    # return any rays that are within detector_range distance from
    # detector_center
    # first get the intersect point by taking the plane (0, 1, 0) and rotating
    # by the proper tilt angle.
    final_rays = []
    # detector_normal_vec = transformLG(1e-10, 1, 1e-10, [0, 0, 0],
    #                                   detector_tilt)
    detector_normal_vec = normal_vec
    for ray in rays:
        detector_intersection, vec = reflect_line_about_plane(
            np.array(ray[2]), np.array(ray[3]), np.array(detector_center),
            np.array(detector_normal_vec))
        distance_from_center = dist(detector_intersection, detector_center)
        # we realistically want this to be a perfect reflection backwards..
        # print('incoming vector = %s' % list(ray[3]))
        # print('"reflected" vector = %s' % list(vec))
        # print('distance from center = %s' % distance_from_center)
        if (distance_from_center <= detector_range):
            new_ray = [ray[0], ray[1], ray[2], ray[3], ray[4]]
            new_ray[4] = ray[4] + dist(ray[2], detector_intersection)
            new_ray[2] = list(detector_intersection)
            final_rays.append(new_ray)

    return final_rays


''' give rays to be run through this FTS at a specific y, returns the good rays and the region. not used anymore
def RunRaysMi(Rays,y): #just give number of rays to be run through this FTS at a specific y!
    Rayf = OFDM(Rays,y)
    G= checkoutraysM(Rayf,[160.375,-113],7.9375) # GOOD RAYS ONLY
    Gtestsorted = sortgrid(G)
    Regions = regionalize(Gtestsorted)
    return Gtestsorted,Regions'''

''' give rays to be run through this FTS at a specific y, returns the good rays and the region. Ends with reflecting of foutput elipsoid. DOES NOT INCLUDE TRAVEL TO DETECTOR.'''


def RunRaysM(Rays, y):  # just give rays to be run through this FTS at a specific y!
    Rayf = OFDM(Rays, y)
    G = checkoutraysM(Rayf, [160.375, -113], 7.9375)  # GOOD RAYS ONLY
    # Gtestsorted = sortgrid(G)
    return G


def run_rays_mirror_second_half(intermediate_rays, y):
    ray_f = ofdm_second_half(intermediate_rays, y)
    g = checkoutraysM(ray_f, [160.375, -113], 20)  # changed from 7.9375
    return g


'''makes n rays with radius r around the fixed source point origin that is the focus of the first ellipsoid. '''


def makeraysiFIXED(n, r):
    sourcepointorigin = [-160.375, -113., -80.0]  # global
    sourcethet = [0., 0., 0.]  # SHOT STRAIGHT UP
    sourcepoints = specsource(r, sourcepointorigin, sourcethet, n)  # SOURCE
    Rays = FSRay(n, sourcepoints, sourcethet, origin10)
    for i in range(n):
        Rays[i][2] = sourcepointorigin
        v1x, v1y, v1z = transformLG(0., 0., 1, originG, sourcethet)
        v2 = sepop(v1x, v1y, v1z)
        Rays[i][3] = v2
    return Rays


'''
def SumjRegionsMi_TestG(Rays,y): #ALSO INCORPORATES PHASE
    OutRays,regions=RunRaysMi(Rays,y)
    Regions = list(regions)
    jx,jy = gridlines(7.9375,[160.375,-113],200)
    # possible wavelengths (30-300 Ghz), steps of 1Ghz
    LamdAll = np.linspace(1, 10,300)
    DetTot = []
    for j in range(len(regions)):
        ExTot = []
        EyTot = []
        for i in range(len(Regions[j])): #All rays in region j
            # ExTot = []
            # EyTot = []
            if i != 0 and i != 1:
                JRegion = Regions[j]
                o,p = JRegion[0],JRegion[1] # jx and jy defining the jth region
                Raym = OutRays[JRegion[i]] #ith ray in the jth region
                if o != len(jx)-1 and p !=len(jy)-1:
                    # w = 1
                    w = gaussian3d(Raym[2][0],Raym[2][1],.4,
                                   (jx[o]+jx[o+1])/2,(jy[p]+jy[p+1])/2)
                else:
                    w = 0 #(skipping gaussian)
                # Raym[1] is intensity!!! #split into x and y components of AMPLITUDE field
                # w = 1
                I = Raym[1]
                thet = Raym[0]
                # multiplied by gaussian
                Ex1,Ey1 = w*np.sqrt(I)*np.cos(thet),w*np.sqrt(I)*np.sin(thet)
                # only one frequency
                Lamd = 3.3
                phase = np.exp(1j*(Raym[4]%Lamd)*2*np.pi/Lamd)
                # phase = np.exp(2*np.pi*1j*Raym[4]/Lamd)
                Ex = Ex1*phase
                Ey = Ey1*phase
                ExTot.append(Ex)
                EyTot.append(Ey)
        Ij = (np.sum(ExTot)*np.sum(ExTot).conjugate()) + \
              (np.sum(EyTot)*np.sum(EyTot).conjugate())
        DetTot.append(Ij.real)
    return np.sum(DetTot)'''

'''Tests straight central axis ray '''


def RunFTSLimiStraightTest(n, r, div, Lim):
    Power = []
    Delay = []
    Rays = makeraysiFIXED(n, r)
    for y in np.linspace(-int(Lim), int(Lim), div):
        I = SumjRegionsMi_TestG(Rays, y)
        Power.append(I)
        Delay.append(y)
    return Power, Delay


'''Gaussian normalized '''


# assuming is symmetric, making peak = 1
def Airygaussian3dNORM(x, y, sig, mux, muy):
   # A = 1
    A = (1/((sig**3)*(2*np.pi)**(3/2)))
    return A*np.exp(-(((x-mux)**2)/(2*sig**2) + ((y-muy)**2)/(2*sig**2)))


'''def Airygaussian3d(x,y,sig,mux,muy): #assuming is symmetric, making peak = 1
    A = 1
    # A = (1/((sig**3)*(2*np.pi)**(3/2)))
    return A*np.exp(-(((x-mux)**2)/(2*sig**2) + ((y-muy)**2)/(2*sig**2)))'''

'''given LAST ray and its wavelength, return sig, mux and muy to then be used in gaussian. see pg 79 for more details(approx of airy func).'''


def MakeGaussian(Ray, Lamd):
    mux, muy = Ray[2][0], Ray[2][1]  # center of gaussian is intersection point
    width = 3.0988*Lamd
    sig = width/3
    return sig, mux, muy


'''#given the two positions of the last rays and wavelength, returns percentage of overlap (out of 1)
def gaussoverlap(Ray1,Ray2,Lamd):
    sig1,mux1,muy1 = MakeGaussian(Ray1,Lamd)
    sig2,mux2,muy2 = MakeGaussian(Ray2,Lamd)
    p1 = [mux1,muy1,80.] #points in 3d in GLOBAL coordinates
    p2 = [mux2,muy2,80.]
    MDValue = dist(p1,p2)
    MD = [0,.25,.5,.75,1,1.1,1.2,1.3,1.5,1.7,1.9,
        2.1,2.3,2.6,2.9,3.4,3.7,3.9] #mean difference
    # Gaussian Percent
    GP = [1,.9,.8,.7,.62,.6,.55,.5,.45,.4,.35,.3,.25,.2,.15,.1,.07,.05]
    # index number in array for closest value
    idx = (np.abs(MD-MDValue)).argmin()
    return GP[idx]'''

# find CENTER of each pixel now as [x,y]


def MakePixels(jx, jy):
    pix = []
    for o in range(len(jx)-1):
        for p in range(len(jy)-1):
            r = 7.9375
            xpix, ypix = (jx[o]+jx[o+1])/2, (jy[p]+jy[p+1])/2
            d = np.sqrt((xpix-160.375)**2 + ((ypix-(-113))**2))
            if d <= r:
                pix.append([xpix, ypix])
    return pix


def get_grid_attrs(gridlines):
    x_grid, y_grid = gridlines
    x_range = x_grid[-1] - x_grid[0]
    y_range = y_grid[-1] - y_grid[0]
    num = len(x_grid) - 1
    x_div = x_range / num
    y_div = y_range / num
    return x_div, y_div, x_grid[0], y_grid[0], num

# make a function to map a coordinate to a bin
# then also make a function to calculate area of each bin to normalize power


def get_bin(point, gridlines):
    # Get the grid increments which make up our bins
    x_div, y_div, x_0, y_0, num = get_grid_attrs(gridlines)

    # Calculate the bins simply as the number of divisions we have gone across
    # the detector
    bin_x = (point[0] - x_0) // x_div
    bin_y = (point[1] - y_0) // y_div

    # Make sure this point lies in the bin range
    assert bin_x < num and bin_y < num

    return bin_y, bin_y


def inside_detectorQ(point, det_center, det_radius):
    dist = np.sqrt(np.sum((point - det_center) ** 2))
    return (dist < det_radius)


def get_rand_points(x_min, x_max, y_min, y_max, num):
    rand_x = (x_max - x_min) * np.random.sample(num) + x_min
    rand_y = (y_max - y_min) * np.random.sample(num) + y_min
    return np.array([rand_x, rand_y]).T

# quick monte carlo estimate


def calculate_bin_area_percent(gridlines, bin, det_center, det_radius):
    x_grid, y_grid = gridlines
    x_min, x_max = x_grid[bin[0]], x_grid[bin[0] + 1]
    y_min, y_max = y_grid[bin[1]], y_grid[bin[1] + 1]
    print(x_min, x_max)

    # generate random points inside this bin range
    num_points = 10000
    points = get_rand_points(x_min, x_max, y_min, y_max, num_points)
    check_points = np.array([inside_detectorQ(
        point, det_center, det_radius) for point in points])
    frac = (np.sum(check_points) / len(check_points))
    return frac


def bin_detector_points(final_rays, num_bins, det_center, det_radius):
    bin_gridlines = gridlines(det_radius, det_center, num_bins)
    bin_array = np.zeros((int(np.sqrt(num_bins)), int(np.sqrt(num_bins))))
    for mirror_pos in range(len(det)):
        for point in det[mirror_pos]:
            # don't care about z coordinate
            bin_x, bin_y = get_bin(point[:-1], bin_gridlines)
            bin_array[bin_x][bin_y] += 1

    return bin_array


''' makes n rays with source points chosen from a random circle with radius r around sourcepointorigin. If r=0, all source points are the given sourcepointorigin. All are launched vertically. '''


def makeraysVERTICAL(sourcepointorigin, r, n):
    sourcethet = [0., 0., 0.]  # SHOT STRAIGHT UP
    sourcepoints = specsource(r, sourcepointorigin, sourcethet, n)  # SOURCE
    Rays = FSRay(n, sourcepoints, sourcethet, origin10)
    for i in range(n):
        v1x, v1y, v1z = transformLG(0., 0., 1, originG, sourcethet)
        v2 = sepop(v1x, v1y, v1z)
        Rays[i][3] = v2
    return Rays


'''Same as above with initial phase all set to zero, rather than random phase. '''


def makeraysVERTICAL_Zero(sourcepointorigin, r, n):
    sourcethet = [0., 0., 0.]  # SHOT STRAIGHT UP
    sourcepoints = specsource(r, sourcepointorigin, sourcethet, n)  # SOURCE
    Rays = FSRay_Zero(n, sourcepoints, sourcethet, origin10)
    for i in range(n):
        v1x, v1y, v1z = transformLG(0., 0., 1, originG, sourcethet)
        v2 = sepop(v1x, v1y, v1z)
        Rays[i][3] = v2
    return Rays


'''Runs one ray center axis with pixel detector and airy pattern interference. '''


def RunOneRay(Lamd, Nsize, spo):
    n = 1
    r = 0
    Rays = makeraysVERTICAL(spo, r, n)
    # these are now the PIXELS
    jx, jy = gridlines(7.9375, [160.375, -113], 200)
    Pix = MakePixels(jx, jy)  # center of each pixel
    Ij = []
    Delay = []
    for y in np.linspace(-18, 18, int(Nsize)):
        PTot = 0
        OutRays = RunRaysM(Rays, y)  # eight each
        # Overlap = gaussoverlap(OutRays[0],OutRays[5],3.3) #two paths that hit two different spots
        for j in range(len(Pix)):  # per PIXEL
            Ex4i = 0  # adding PER PIXEL from parts of RAYS in this PIXEL
            Ey4i = 0  # THIS IS WHERE THEY WILL INTERFERE
            for i in range(len(OutRays)):  # per ray IN THIS PIXEL
                I = OutRays[i][1]
                thet = OutRays[i][0]
                # factor of 2??
                phase = np.exp(1j*(OutRays[i][4]*2*np.pi/Lamd))
                Ex1 = np.sqrt(I)*np.cos(thet)
                Ey1 = np.sqrt(I)*np.sin(thet)
                Ex = Ex1*phase
                Ey = Ey1*phase
                # doing summation over entire detector
                sig, mux, muy = MakeGaussian(OutRays[i], Lamd)
                Gr = Airygaussian3dNORM(Pix[j][0], Pix[j][1], sig, mux, muy)
                Ex4i = Ex4i + Gr*Ex
                Ey4i = Ey4i + Gr*Ey
            PTot = PTot + (Ex4i*Ex4i.conjugate()).real + \
                (Ey4i*Ey4i.conjugate()).real
        Delay.append(y*0.95630475596*4)
        Ij.append(PTot)
    return Delay, Ij


'''Making the source out of many different rays on a source with pixel detector
and airy pattern interference '''


def RunSource(Lamd, Nsize, spo, n, r):
    Rays = makerays(spo, thetG, r, n)  # sourcethet as [0,0,0]
    # these are now the PIXELS
    jx, jy = gridlines(7.9375, [160.375, -113], 200)
    Pix = MakePixels(jx, jy)  # center of each pixel
    Ij = []
    Delay = []
    for y in np.linspace(-18, 18, int(Nsize)):
        PTot = 0
        OutRays = RunRaysM(Rays, y)  # eight each
        for j in range(len(Pix)):  # per PIXEL
            Ex4i = 0  # adding PER PIXEL from parts of RAYS in this PIXEL
            Ey4i = 0  # THIS IS WHERE THEY WILL INTERFERE
            for i in range(len(OutRays)):  # per ray IN THIS PIXEL
                I = OutRays[i][1]
                thet = OutRays[i][0]
                # factor of 2??
                phase = np.exp(1j*(OutRays[i][4]*2*np.pi/Lamd))
                Ex1 = np.sqrt(I)*np.cos(thet)
                Ey1 = np.sqrt(I)*np.sin(thet)
                Ex = Ex1*phase
                Ey = Ey1*phase
                # doing summation over entire detector
                sig, mux, muy = MakeGaussian(OutRays[i], Lamd)
                Gr = Airygaussian3dNORM(Pix[j][0], Pix[j][1], sig, mux, muy)
                Ex4i = Ex4i + Gr*Ex
                Ey4i = Ey4i + Gr*Ey
            PTot = PTot + (Ex4i*Ex4i.conjugate()).real + \
                (Ey4i*Ey4i.conjugate()).real
        Delay.append(y*0.95630475596*4)
        Ij.append(PTot)
    return Delay, Ij


'''Makes n rays. sourcepoints random in radius r around
sourcepointorigin. random launch angles in hemisphere centered around
sourcethet. '''


def makerays(sourcepointorigin, sourcethet, r, n):
    sourcepoints = specsource(r, sourcepointorigin, sourcethet, n)  # SOURCE
    Rays = FSRay(n, sourcepoints, sourcethet, origin10)
    return Rays


'''Makes n rays. sourcepoints random in radius r around
sourcepointorigin. random launch angles in hemisphere centered around
sourcethet and < lim (solid angle in radians) launch angle. '''


def makerays_narrow(sourcepointorigin, sourcethet, r, n, lim):
    sourcepoints = specsource(r, sourcepointorigin, sourcethet, n)  # SOURCE
    Rays = FSRay_narrow(n, sourcepoints, sourcethet, origin10, lim)
    return Rays


'''Same as above but with initial phase of zero instead of random. '''


def makerays_Zero(sourcepointorigin, sourcethet, r, n):
    sourcepoints = specsource(r, sourcepointorigin, sourcethet, n)  # SOURCE
    Rays = FSRay_Zero(n, sourcepoints, sourcethet, origin10)
    return Rays


'''Same as above but with initial phase of zero instead of random and < lim
(solid angle in radians) launch angle. '''


def makerays_Zero_narrow(sourcepointorigin, sourcethet, r, n, lim):
    sourcepoints = specsource(r, sourcepointorigin, sourcethet, n)  # SOURCE
    Rays = FSRay_Zero_narrow(n, sourcepoints, sourcethet, origin10, lim)
    return Rays


'''Calculates number of samples needed for a given wavelength (in mm) '''


def Nsized(Lamd, ymax=18):
    L = 2 * ymax * 0.95630475596 * 4  # Length of the detector scan
    Ks = 2*np.pi/Lamd  # K value
    Nmin = math.ceil(L*Ks)  # L * K
    n = math.log(Nmin, 2)  # log (L * K)
    n = n+1
    return math.ceil(2**n)


def Maxf(x, y):
    maxy = max(y)
    # Find the x value corresponding to the maximum y value
    maxx = x[y.argmax()]
    return 300*maxx, maxy, y.argmax()


'''Runs the simulation with 1 ray down the center axis. power is simulated
without pixels or airy pattern.'''


def RunOneRay_nopix(Lamd, Nsize, spo):  # no pixels
    n = 1
    r = 0
    Rays = makeraysVERTICAL(spo, r, n)
    # jx,jy = gridlines(7.9375,[160.375,-113],200) #these are now the PIXELS
    # Pix = MakePixels(jx,jy) #center of each pixel
    Ij = []
    Delay = []
    # nsize being number of positions of mirror
    for y in np.linspace(-18, 18, int(Nsize)):
        PTot = 0
        OutRays = RunRaysM(Rays, y)  # eight each
        Ext = 0
        Eyt = 0
        for i in range(len(OutRays)):  # per ray IN THIS PIXEL
            I = OutRays[i][1]
            thet = OutRays[i][0]
            phase = np.exp(1j*(OutRays[i][4]*2*np.pi/Lamd))  # factor of 2??
            Ex1 = np.sqrt(I)*np.cos(thet)
            Ey1 = np.sqrt(I)*np.sin(thet)
            Ex = Ex1*phase
            Ey = Ey1*phase
            Ext = Ext + Ex
            Eyt = Eyt + Ey
        PTot = PTot + (Ext*Ext.conjugate()).real + (Eyt*Eyt.conjugate()).real
        Delay.append(y*0.95630475596*4)
        Ij.append(PTot)
    return Delay, Ij


'''Simulation of interference of probability function of a single photon. 500
rays with initial phase of zero from a single source point, random launch
points, and power is summed before squared. To show Chamberlain loss (large
etendue) '''


def RunRays_Prob(Lamd, Nsize, spo):
    n = 500
    r = 0
    thetG = [0, 0, 0]
    # Rays = makeraysVERTICAL(spo,r,n)
    Rays = makerays_Zero(spo, thetG, r, n)
    # jx,jy = gridlines(7.9375,[160.375,-113],200) #these are now the PIXELS
    # Pix = MakePixels(jx,jy) #center of each pixel
    Ij = []
    Delay = []
    for y in np.linspace(-18, 18, int(Nsize)):
        PTot = 0
        OutRays = RunRaysM(Rays, y)  # eight each
        # Overlap = gaussoverlap(OutRays[0],OutRays[5],3.3) #two paths that hit two different spots
        # for j in range(len(Pix)): #per PIXEL
        for j in range(1):
            Ex4i = 0  # adding PER PIXEL from parts of RAYS in this PIXEL
            Ey4i = 0  # THIS IS WHERE THEY WILL INTERFERE
            for i in range(len(OutRays)):  # per ray IN THIS PIXEL
                I = OutRays[i][1]
                thet = OutRays[i][0]
                # factor of 2??
                phase = np.exp(1j*(OutRays[i][4]*2*np.pi/Lamd))
                Ex1 = np.sqrt(I)*np.cos(thet)
                Ey1 = np.sqrt(I)*np.sin(thet)
                Ex = Ex1*phase
                Ey = Ey1*phase
                # doing summation over entire detector
                # sig,mux,muy = MakeGaussian(OutRays[i],Lamd)
                # Gr = Airygaussian3dNORM(Pix[j][0],Pix[j][1],sig,mux,muy)
                Gr = 1
                Ex4i = Ex4i + Gr*Ex
                Ey4i = Ey4i + Gr*Ey
            PTot = PTot + (Ex4i*Ex4i.conjugate()).real + \
                (Ey4i*Ey4i.conjugate()).real
        Delay.append(y*0.95630475596*4)
        Ij.append(PTot)
    return Delay, Ij


'''Simulation of interference of probability function of a single photon. 500
rays with initial phase of zero from a single source point, random launch
points, and power is summed before squared. To show Chamberlain loss (large
etendue) '''


def RunRays_Prob_narrow(Lamd, Nsize, spo, lim):
    n = 500
    r = 0
    thetG = [0, 0, 0]
    # Rays = makeraysVERTICAL(spo,r,n)
    # lim is solid angle wanted in radians
    Rays = makerays_Zero_narrow(spo, thetG, r, n, lim)
    # jx,jy = gridlines(7.9375,[160.375,-113],200) #these are now the PIXELS
    # Pix = MakePixels(jx,jy) #center of each pixel
    Ij = []
    Delay = []
    for y in np.linspace(-18, 18, int(Nsize)):
        PTot = 0
        # all rays that made it through the detector
        OutRays = RunRaysM(Rays, y)
        # Overlap = gaussoverlap(OutRays[0],OutRays[5],3.3) #two paths that hit two different spots
        # for j in range(len(Pix)): #per PIXEL
        for j in range(1):  # no pixels
            Ex4i = 0  # adding PER PIXEL from parts of RAYS in this PIXEL
            Ey4i = 0  # THIS IS WHERE THEY WILL INTERFERE
            for i in range(len(OutRays)):  # per ray IN THIS PIXEL
                I = OutRays[i][1]  # amplitude
                thet = OutRays[i][0]  # polarization
                # e^ix2pi/lambda, x = distance traveleld
                phase = np.exp(1j*(OutRays[i][4]*2*np.pi/Lamd))
                Ex1 = np.sqrt(I)*np.cos(thet)  # polarization
                Ey1 = np.sqrt(I)*np.sin(thet)
                Ex = Ex1*phase  # phase
                Ey = Ey1*phase
                # doing summation over entire detector
                # sig,mux,muy = MakeGaussian(OutRays[i],Lamd)
                # Gr = Airygaussian3dNORM(Pix[j][0],Pix[j][1],sig,mux,muy)
                Gr = 1
                Ex4i = Ex4i + Gr*Ex  # add electric fields of all rays
                Ey4i = Ey4i + Gr*Ey
            PTot = PTot + (Ex4i*Ex4i.conjugate()).real + \
                (Ey4i*Ey4i.conjugate()).real
        Delay.append(y*0.95630475596*4)
        Ij.append(PTot)
    return Delay, Ij


'''Givin initial rays, just change source (set shift of source point) keeping
everything else identical '''


def makerays_Zero_narrow_SetShift(sourcepoint, Rays):
    spT = [sourcepoint[0], sourcepoint[1], sourcepoint[2]]
    for i in range(len(Rays)):
        Rays[i][2] = spT
    return Rays


'''Simulation of interference of probability function of a single photon. 500
rays with initial phase of zero from a single source point, random launch
points, and power is summed before squared. To show Chamberlain loss (large
etendue) while moving source and giving initial set of rays'''


def RunRays_Prob_narrow_SetRays(Lamd, Nsize, spo, lim, Rays, ymax=18):
    # n = 500
    # r = 0
    thetG = [0, 0, 0]
    # Rays = makeraysVERTICAL(spo,r,n)
    # Rays = makerays_Zero_narrow(spo,thetG,r,n,lim) #lim is solid angle wanted in radians
    # jx,jy = gridlines(7.9375,[160.375,-113],200) #these are now the PIXELS
    # Pix = MakePixels(jx,jy) #center of each pixel
    Ij = []
    Delay = []
    DETECTOR = []  # added to capture final points on detector
    for y in np.linspace(-1 * ymax, ymax, int(Nsize)):
        PTot = 0
        OutRays = RunRaysM(Rays, y)  # all rays that made it to the detector
        # print('finished running rays through simulation. Now summing power.')
        # Overlap = gaussoverlap(OutRays[0],OutRays[5],3.3) #two paths that hit two different spots
        # for j in range(len(Pix)): #per PIXEL
        P = []  # final points on detector.
        for j in range(1):  # no pixels
            Ex4i = 0  # adding PER PIXEL from parts of RAYS in this PIXEL
            Ey4i = 0  # THIS IS WHERE THEY WILL INTERFERE
            for i in range(len(OutRays)):  # per ray IN THIS PIXEL
                I = OutRays[i][1]  # amplitude
                thet = OutRays[i][0]  # polarization
                # e^ix2pi/lambda, x = distance traveleld
                phase = np.exp(1j*(OutRays[i][4]*2*np.pi/Lamd))
                Ex1 = np.sqrt(I)*np.cos(thet)  # polarization
                Ey1 = np.sqrt(I)*np.sin(thet)
                Ex = Ex1*phase  # phase
                Ey = Ey1*phase
                # doing summation over entire detector
                # sig,mux,muy = MakeGaussian(OutRays[i],Lamd)
                # Gr = Airygaussian3dNORM(Pix[j][0],Pix[j][1],sig,mux,muy)
                Gr = 1
                Ex4i = Ex4i + Gr*Ex  # add electric fields of all rays
                Ey4i = Ey4i + Gr*Ey
                # Just return the full ray instead for more info
                P.append(OutRays[i])
                # P.append(OutRays[i][2])  # spot on detector.
            PTot = PTot + (Ex4i*Ex4i.conjugate()).real + \
                (Ey4i*Ey4i.conjugate()).real
        Delay.append(y*0.95630475596*4)
        Ij.append(PTot)
        DETECTOR.append(P)
    return Delay, Ij, DETECTOR


def run_rays_prob_narrow_vectorized(wavelength, num_mirror_positions, spo, lim,
                                    rays, ymax=18):
    # n = 500
    # r = 0
    # thetG = [0, 0, 0]
    # Rays = makeraysVERTICAL(spo,r,n)
    # lim is solid angle wanted in radians
    # Rays = makerays_Zero_narrow(spo,thetG,r,n,lim)
    # jx,jy = gridlines(7.9375,[160.375,-113],200) #these are now the PIXELS
    # Pix = MakePixels(jx,jy) #center of each pixel
    ij = []
    delay = []
    detector = []  # added to capture final points on detector

    # Create a lookup table with information about locations of each of the
    # rays after the first 4 reflections since these are the same every time
    intermediate_rays = ofdm_first_half(rays)

    # Then for each y perform the rest of the raytracing
    for y in tqdm(np.linspace(-1 * ymax, ymax, int(num_mirror_positions))):
        # all rays that made it to the detector
        outrays = run_rays_mirror_second_half(intermediate_rays, y)
        # print('finished running rays through simulation. Now summing power.')
        # two paths that hit two different spots
        # Overlap = gaussoverlap(OutRays[0],OutRays[5],3.3)
        # for j in range(len(Pix)): #per PIXEL
        total_power = sum_power_vectorized(outrays, wavelength)

        delay.append(y*0.95630475596*4)  # should get rid of this magic number
        ij.append(total_power)
        detector.append(outrays)
    return delay, ij, detector


def sum_power_vectorized(outrays, wavelength):
    outrays = np.array(outrays, dtype='object')
    intensity = np.asarray(outrays[:, 1], dtype='complex128')
    theta = np.asarray(outrays[:, 0], dtype='complex128')
    distance = np.asarray(outrays[:, 4], dtype='complex128')
    phase = np.exp(1j * ((distance) * 2 * np.pi / wavelength))
    ex1 = np.sqrt(intensity) * np.cos(theta)  # polarization
    ey1 = np.sqrt(intensity) * np.sin(theta)
    ex = ex1 * phase  # phase
    ey = ey1 * phase
    # doing summation over entire detector
    # sig,mux,muy = MakeGaussian(OutRays[i],Lamd)
    # Gr = Airygaussian3dNORM(Pix[j][0],Pix[j][1],sig,mux,muy)
    gr = 1
    ex = gr * ex  # add electric fields of all rays
    ey = gr * ey

    sum_ex = ex.sum()
    sum_ey = ey.sum()

    return (sum_ex * sum_ex.conjugate() + sum_ey * sum_ey.conjugate()).real


def sum_power_matrix(outrays, wavelength):
    theta = np.asarray(outrays[:, 0], dtype='complex64')
    intensity = np.asarray(outrays[:, 1], dtype='complex64')
    distance = np.asarray(outrays[:, 2], dtype='complex64')
    phase = np.exp(1j * (distance * 2 * np.pi / wavelength))
    ex1 = np.sqrt(intensity) * np.cos(theta)  # polarization
    ey1 = np.sqrt(intensity) * np.sin(theta)
    ex = ex1 * phase  # phase
    ey = ey1 * phase
    # doing summation over entire detector
    # sig,mux,muy = MakeGaussian(OutRays[i],Lamd)
    # Gr = Airygaussian3dNORM(Pix[j][0],Pix[j][1],sig,mux,muy)
    gr = 1
    ex = gr * ex  # add electric fields of all rays
    ey = gr * ey

    sum_ex = ex.sum()
    sum_ey = ey.sum()

    return (sum_ex * sum_ex.conjugate() + sum_ey * sum_ey.conjugate()).real


def get_interferogram(outrays, wavelength):
    theta = outrays[:, :, 0]
    intensity = outrays[:, :, 1]
    distance = outrays[:, :, 2]

    phase = np.exp(1j * (distance * 2 * np.pi / wavelength))
    ex1 = np.sqrt(intensity) * np.cos(theta)
    ey1 = np.sqrt(intensity) * np.sin(theta)

    ex = ex1 * phase
    ey = ey1 * phase

    return np.square(np.abs((ex.sum(axis=1)))) + np.square(
        np.abs((ey.sum(axis=1))))


def run_rays_prob_narrow_return_rays(num_mirror_positions, rays, ymax=18):
    delay = []
    detector = []  # added to capture final points on detector

    # Create a lookup table with information about locations of each of the
    # rays after the first 4 reflections since these are the same every time
    # (saves a factor of 2 in the calculation)
    intermediate_rays = ofdm_first_half(rays)

    # Then for each y perform the rest of the raytracing
    for y in tqdm(np.linspace(-1 * ymax, ymax, int(num_mirror_positions))):
        # all rays that made it to the detector
        outrays = run_rays_mirror_second_half(intermediate_rays, y)

        delay.append(y*0.95630475596*4)  # should get rid of this magic number
        detector.append(outrays)

    return delay, detector


def test_optimized_function(n_rays=20, n_mirror_positions=200,
                            return_elems=False, time_funcs=True):
    print('running functions...')
    wavelength = 143.
    solid_angle_limit = np.pi / 9
    rays = makerays_Zero_narrow([0., 0., 0.], [0., 0., 0.], 0., n_rays,
                                solid_angle_limit)
    source_point_origin = [-160.375, -113., -80.]
    rays_shifted = makerays_Zero_narrow_SetShift(source_point_origin, rays)
    # Now run these rays using both methods and see that they return the same
    # results
    args = [wavelength, n_mirror_positions, source_point_origin,
            solid_angle_limit, rays_shifted]
    if (time_funcs):
        start = time.time()
    delay1, ij1, det1 = RunRays_Prob_narrow_SetRays(*args)
    if (time_funcs):
        time1 = time.time() - start
        print('time to run unoptimized func = %.3g minutes' % (time1 / 60))
    if (time_funcs):
        start = time.time()
    delay2, ij2, det2 = run_rays_prob_narrow_vectorized(*args)
    if (time_funcs):
        time2 = time.time() - start
        diff = time1 / time2
        print('time to run optimized func = %.3g minutes' % (time2 / 60))
        print('optimized function is %.3g times faster!' % diff)
    # Make sure delays are the same
    print('checking outputs...')
    assert (np.array(delay1) == np.array(delay2)).all()
    assert np.abs((np.array(ij1) - np.array(ij2)).sum()) < .000001

    for mirror_pos in range(len(det1)):
        for ray1, ray2 in zip(det1[mirror_pos], det2[mirror_pos]):
            for i in range(len(ray1)):
                assert (np.abs(
                    (np.array(ray1[i]) - np.array(ray2[i])).sum()) < .000001)

    print('Outputs match. Completed.')
    if (return_elems):
        return delay1, ij1, det1, delay2, ij2, det2
    return


def test_return_rays(n_rays=20, n_mirror_positions=200):
    print('running functions...')
    wavelength = 150 * np.random.random() + 50
    print('testing over wavelength of %s' % wavelength)
    solid_angle_limit = np.pi / 9
    rays = makerays_Zero_narrow([0., 0., 0.], [0., 0., 0.], 0., n_rays,
                                solid_angle_limit)
    source_point_origin = [-160.375, -113., -80.]
    rays_shifted = makerays_Zero_narrow_SetShift(source_point_origin, rays)
    # Now run these rays using both methods and see that they return the same
    # results
    args = [wavelength, n_mirror_positions, source_point_origin,
            solid_angle_limit, rays_shifted]
    delay1, ij1, det1 = run_rays_prob_narrow_vectorized(*args)
    delay2, det2 = run_rays_prob_narrow_return_rays(
        n_mirror_positions, rays_shifted)

    # Make sure delays are the same
    print('checking outputs...')
    assert (np.array(delay1) == np.array(delay2)).all()

    for i, y_rays in enumerate(det2):
        assert np.abs(
            sum_power_vectorized(y_rays, wavelength) - ij1[i]) < .00001

    print('Outputs match. Completed.')
    return


def drawcircle(h, k, r):
    x = np.linspace(h-r, h+r, 100)
    ypos = []
    yneg = []
    for i in range(100):
        ypos.append(k + np.sqrt(r**2 - (x[i]-h)**2))
        yneg.append(k - np.sqrt(r**2 - (x[i]-h)**2))
    return x, ypos, yneg


'''Runs one ray through the simulation, random phase, random launch angle. Also
includes number of rays for geometric loss. to show geometric loss of
power. '''


def Part1(Lamd, Nsize, spo):
    n = 1
    r = 0
    thetG = [0, 0, 0]
    Rays = makerays(spo, thetG, r, n)
    Ij = []
    Delay = []
    # number of rays that hit detector as function of mirror position (y)
    Number = []
    for y in np.linspace(-18, 18, int(Nsize)):
        PTot = 0
        OutRays = RunRaysM(Rays, y)  # eight each if n =1
        if len(OutRays) == 0:
            PTot = 0
        else:
            Ex4i = 0
            Ey4i = 0
            for i in range(len(OutRays)):  # per ray that hit detector
                I = OutRays[i][1]
                thet = OutRays[i][0]
                # factor of 2??
                phase = np.exp(1j*(OutRays[i][4]*2*np.pi/Lamd))
                Ex1 = np.sqrt(I)*np.cos(thet)
                Ey1 = np.sqrt(I)*np.sin(thet)
                Ex = Ex1*phase
                Ey = Ey1*phase
                Gr = 1
                Ex4i = Ex4i + Gr*Ex
                Ey4i = Ey4i + Gr*Ey
            PTot = PTot + (Ex4i*Ex4i.conjugate()).real + \
                (Ey4i*Ey4i.conjugate()).real
        Delay.append(y*0.95630475596*4)
        Ij.append(PTot)
        Number.append(len(OutRays))
    return Delay, Ij, Number


'''runs n rays individually. '''


def RunNRays_NoPix(Lamd, Nsize, spo, n):  # multiple individual rays.
    Ij = np.zeros(Nsize)  # empty array of proper size
    Numz = np.zeros(Nsize)
    for i in range(int(n)):
        Delay, Pow1, Num = Part1(Lamd, Nsize, spo)
        Numz = Numz+np.array(Num)
        Ij = Ij + np.array(Pow1)
    return Delay, Ij, Numz


''' Number of rays that reach detector as a function of mirror location '''


def Part2(Lamd, Nsize, spo):
    n = 1000
    r = 0
    thetG = [0, 0, 0]
    Rays = makerays(spo, thetG, r, n)
    N = []
    # number of rays that hit detector as function of mirror position (y)
    Number = []
    for y in np.linspace(-18, 18, int(Nsize)):
        PTot = 0
        OutRays = RunRaysM(Rays, y)  # eight each if n =1
        N.append(len(OutRays))
    return N


'''This is the function that runs the simulation with n rays, of wavelength
lamd, with Nsize sample points (mirror positions from -18 to 18) a
sourcepointorigin at spo. Returns a single array of all of the rays generated
at every instance '''


def RunRays_ToPickle(Lamd, Nsize, spo, n):  # no pixels
    # n = 1
    r = 0
    Rays = makerays_Zero(spo, thetG, r, n)
    Ij = []
    Delay = []
    Rayf = [[[]for j in range(Nsize+1)] for i in range(n)]
    for k in range(n):
        Rayf[k][0].append('Ray: '+str(k))
    yn = 1
    # nsize being number of positions of mirror
    for y in np.linspace(-18, 18, int(Nsize)):
        for i in range(len(Rays)):
            Paths = [TTTTioMPickle, RRRRioMPickle, TTRRioMPickle, RTTRioMPickle,
                     RTRTioMPickle, TRRTioMPickle, RRTTioMPickle, TRTRioMPickle]
            Ri = Rays[i]
            for j in range(8):
                origin = (0, y, 0)
                if j == 0:
                    Rayf[i][yn].append('Mirror position: '+str(origin))
                out = Paths[j](Ri, p1, p2, p3, p4, origin)
                out = ((Paths[j].__name__)[:-3],)+out
                Rayf[i][yn].append(out)
        yn = yn+1
    return Rayf


''' Below are functions included in PossiblePaths.py'''
''' Entire run of the simulation (from input ray to the output ellipsoid, and
all those below require the position of the mirror (as an origin). Only the
eight paths that reach the detector are included, with each function referring
to fork in the path chosen. For example, TTTTioM is the path of the ray that
was transmitted through all four polarizers with the mirror at the position
'originM', and RTTRioM is the path of the ray that was reflected from polarizer
1, transmitted through polarizer 2 and 3, and and reflected from polarizer 4
with the mirror at the position 'originM'.'''


def TTTTioM(Ri, p1, p2, p3, p4, originM):
    Ray1 = ReflEll(Ri, thet10, origin10, coeffellipse7,
                   center10, range10)  # first ellipsoid
    Ray_TP1 = IntPolT2(Ray1, coeffpolar, originpolar1, p1)  # P1
    Ray_E8 = ReflEll(Ray_TP1, thet6, origin8,
                     coeffellipse56, center8, range8)  # E8
    Ray_TP2 = IntPolT2(Ray_E8, coeffpolar, originpolar2, p2)  # P2
    Ray_E3 = ReflEll(Ray_TP2, thet, origin3,
                     coeffellipse, center3, range3)  # E3
    Ray_M0 = IntM2(Ray_E3, coeffmirr, originM)  # off mirror
    Ray_E4 = ReflEll(Ray_M0, thet, origin4, coeffellipse,
                     center4, range4)  # off E4
    Ray_TP3 = IntPolT2(Ray_E4, coeffpolar, originpolar3, p3)  # P3
    Ray_E5 = ReflEll(Ray_TP3, thet5, origin5, coeffellipse56,
                     center5, range5)  # OFF E5
    Ray_TP4 = IntPolT2(Ray_E5, coeffpolar, originpolar4, p4)
    Ray_E72 = ReflEll(Ray_TP4, thet7, origin7, coeffellipse7,
                      center7, range7)  # OFF E7
    return Ray_E72


def RRRRioM(Ri, p1, p2, p3, p4, originM):
    Ray1 = ReflEll(Ri, thet10, origin10, coeffellipse7, center10, range10)
    Ray_RP1 = IntPolR2(Ray1, coeffpolar, originpolar1, p1)  # p1
    Ray_E9 = ReflEll(Ray_RP1, thet5, origin9,
                     coeffellipse56, center9, range9)  # E9
    Ray_RP2 = IntPolR2(Ray_E9, coeffpolar, originpolar2, p2)  # P2
    Ray_E3 = ReflEll(Ray_RP2, thet, origin3,
                     coeffellipse, center3, range3)  # E3
    Ray_M0 = IntM2(Ray_E3, coeffmirr, originM)  # off mirror
    Ray_E4 = ReflEll(Ray_M0, thet, origin4, coeffellipse,
                     center4, range4)  # off E4
    Ray_RP3 = IntPolR2(Ray_E4, coeffpolar, originpolar3, p3)  # P3
    Ray_E6 = ReflEll(Ray_RP3, thet6, origin6, coeffellipse56,
                     center6, range6)  # off E6
    Ray_RP4 = IntPolR2(Ray_E6, coeffpolar, originpolar4, p4)
    Ray_E72 = ReflEll(Ray_RP4, thet7, origin7, coeffellipse7, center7, range7)
    return Ray_E72


def TTRRioM(Ri, p1, p2, p3, p4, originM):
    Ray1 = ReflEll(Ri, thet10, origin10, coeffellipse7, center10, range10)
    Ray_TP1 = IntPolT2(Ray1, coeffpolar, originpolar1, p1)  # P1
    Ray_E8 = ReflEll(Ray_TP1, thet6, origin8,
                     coeffellipse56, center8, range8)  # E8
    Ray_TP2 = IntPolT2(Ray_E8, coeffpolar, originpolar2, p2)  # P2
    Ray_E3 = ReflEll(Ray_TP2, thet, origin3,
                     coeffellipse, center3, range3)  # E3
    Ray_M0 = IntM2(Ray_E3, coeffmirr, originM)  # off mirror
    Ray_E4 = ReflEll(Ray_M0, thet, origin4, coeffellipse,
                     center4, range4)  # off E4
    Ray_RP3 = IntPolR2(Ray_E4, coeffpolar, originpolar3, p3)  # P3
    Ray_E6 = ReflEll(Ray_RP3, thet6, origin6, coeffellipse56,
                     center6, range6)  # off E6
    Ray_RP4 = IntPolR2(Ray_E6, coeffpolar, originpolar4, p4)
    Ray_E72 = ReflEll(Ray_RP4, thet7, origin7, coeffellipse7, center7, range7)
    return Ray_E72


def RTTRioM(Ri, p1, p2, p3, p4, originM):
    Ray1 = ReflEll(Ri, thet10, origin10, coeffellipse7, center10, range10)
    Ray_RP1 = IntPolR2(Ray1, coeffpolar, originpolar1, p1)  # P1
    Ray_E9 = ReflEll(Ray_RP1, thet5, origin9,
                     coeffellipse56, center9, range9)  # E9
    Ray_TP2 = IntPolT2(Ray_E9, coeffpolar, originpolar2, p2)  # P2
    Ray_E1 = ReflEll(Ray_TP2, thet, origin1,
                     coeffellipse, center1, range1)  # E1
    Ray_M0 = IntM2(Ray_E1, coeffmirr, originM)  # off mirror
    Ray_E2 = ReflEll(Ray_M0, thet, origin2, coeffellipse,
                     center2, range2)  # off E2
    Ray_TP3 = IntPolT2(Ray_E2, coeffpolar, originpolar3, p3)  # P3
    Ray_E6 = ReflEll(Ray_TP3, thet6, origin6,
                     coeffellipse56, center6, range6)  # E6
    Ray_RP4 = IntPolR2(Ray_E6, coeffpolar, originpolar4, p4)
    Ray_E72 = ReflEll(Ray_RP4, thet7, origin7, coeffellipse7, center7, range7)
    return Ray_E72


def RTRTioM(Ri, p1, p2, p3, p4, originM):
    Ray1 = ReflEll(Ri, thet10, origin10, coeffellipse7, center10, range10)
    Ray_RP1 = IntPolR2(Ray1, coeffpolar, originpolar1, p1)  # P1
    Ray_E9 = ReflEll(Ray_RP1, thet5, origin9,
                     coeffellipse56, center9, range9)  # E9
    Ray_TP2 = IntPolT2(Ray_E9, coeffpolar, originpolar2, p2)  # P2
    Ray_E1 = ReflEll(Ray_TP2, thet, origin1,
                     coeffellipse, center1, range1)  # E1
    Ray_M0 = IntM2(Ray_E1, coeffmirr, originM)  # off mirror
    Ray_E2 = ReflEll(Ray_M0, thet, origin2, coeffellipse,
                     center2, range2)  # off E2
    Ray_RP3 = IntPolR2(Ray_E2, coeffpolar, originpolar3, p3)  # P3
    Ray_E5 = ReflEll(Ray_RP3, thet5, origin5,
                     coeffellipse56, center5, range5)  # E5
    Ray_TP4 = IntPolT2(Ray_E5, coeffpolar, originpolar4, p4)
    Ray_E72 = ReflEll(Ray_TP4, thet7, origin7, coeffellipse7, center7, range7)
    return Ray_E72


def TRRTioM(Ri, p1, p2, p3, p4, originM):
    Ray1 = ReflEll(Ri, thet10, origin10, coeffellipse7, center10, range10)
    Ray_TP1 = IntPolT2(Ray1, coeffpolar, originpolar1, p1)  # P1
    Ray_E8 = ReflEll(Ray_TP1, thet6, origin8,
                     coeffellipse56, center8, range8)  # E8
    Ray_RP2 = IntPolR2(Ray_E8, coeffpolar, originpolar2, p2)  # P2
    Ray_E1 = ReflEll(Ray_RP2, thet, origin1,
                     coeffellipse, center1, range1)  # E3
    Ray_M0 = IntM2(Ray_E1, coeffmirr, originM)  # off mirror
    Ray_E2 = ReflEll(Ray_M0, thet, origin2, coeffellipse,
                     center2, range2)  # off E4
    Ray_RP3 = IntPolR2(Ray_E2, coeffpolar, originpolar3, p3)  # P3
    Ray_E5 = ReflEll(Ray_RP3, thet5, origin5, coeffellipse56, center5, range5)
    Ray_TP4 = IntPolT2(Ray_E5, coeffpolar, originpolar4, p4)
    Ray_E72 = ReflEll(Ray_TP4, thet7, origin7, coeffellipse7, center7, range7)
    return Ray_E72


def RRTTioM(Ri, p1, p2, p3, p4, originM):
    Ray1 = ReflEll(Ri, thet10, origin10, coeffellipse7, center10, range10)
    Ray_RP1 = IntPolR2(Ray1, coeffpolar, originpolar1, p1)  # p1
    Ray_E9 = ReflEll(Ray_RP1, thet5, origin9,
                     coeffellipse56, center9, range9)  # E8
    Ray_RP2 = IntPolR2(Ray_E9, coeffpolar, originpolar2, p2)  # P2
    Ray_E3 = ReflEll(Ray_RP2, thet, origin3,
                     coeffellipse, center3, range3)  # E3
    Ray_M0 = IntM2(Ray_E3, coeffmirr, originM)  # off mirror
    Ray_E4 = ReflEll(Ray_M0, thet, origin4, coeffellipse,
                     center4, range4)  # off E4
    Ray_TP3 = IntPolT2(Ray_E4, coeffpolar, originpolar3, p3)  # P3
    Ray_E5 = ReflEll(Ray_TP3, thet5, origin5, coeffellipse56, center5, range5)
    Ray_TP4 = IntPolT2(Ray_E5, coeffpolar, originpolar4, p4)
    Ray_E72 = ReflEll(Ray_TP4, thet7, origin7, coeffellipse7, center7, range7)
    return Ray_E72


def TRTRioM(Ri, p1, p2, p3, p4, originM):
    Ray1 = ReflEll(Ri, thet10, origin10, coeffellipse7, center10, range10)
    Ray_TP1 = IntPolT2(Ray1, coeffpolar, originpolar1, p1)  # P1
    Ray_E8 = ReflEll(Ray_TP1, thet6, origin8,
                     coeffellipse56, center8, range8)  # E8
    Ray_RP2 = IntPolR2(Ray_E8, coeffpolar, originpolar2, p2)  # P2
    Ray_E1 = ReflEll(Ray_RP2, thet, origin1,
                     coeffellipse, center1, range1)  # E3
    Ray_M0 = IntM2(Ray_E1, coeffmirr, originM)  # off mirror
    Ray_E2 = ReflEll(Ray_M0, thet, origin2, coeffellipse,
                     center2, range2)  # off E4
    Ray_TP3 = IntPolT2(Ray_E2, coeffpolar, originpolar3, p3)  # P3
    Ray_E6 = ReflEll(Ray_TP3, thet6, origin6, coeffellipse56, center6, range6)
    Ray_RP4 = IntPolR2(Ray_E6, coeffpolar, originpolar4, p4)
    Ray_E72 = ReflEll(Ray_RP4, thet7, origin7, coeffellipse7, center7, range7)
    return Ray_E72


def TTTTioM_first_half(Ri, p1, p2, p3, p4):
    Ray1 = ReflEll(Ri, thet10, origin10, coeffellipse7,
                   center10, range10)  # first ellipsoid
    Ray_TP1 = IntPolT2(Ray1, coeffpolar, originpolar1, p1)  # P1
    Ray_E8 = ReflEll(Ray_TP1, thet6, origin8,
                     coeffellipse56, center8, range8)  # E8
    Ray_TP2 = IntPolT2(Ray_E8, coeffpolar, originpolar2, p2)  # P2
    Ray_E3 = ReflEll(Ray_TP2, thet, origin3,
                     coeffellipse, center3, range3)  # E3
    return Ray_E3


# @jit(cache=True)
def TTTTioM_second_half(Ri, p1, p2, p3, p4, originM):
    Ray_M0 = IntM2(Ri, coeffmirr, originM)  # off mirror
    Ray_E4 = ReflEll(Ray_M0, thet, origin4, coeffellipse,
                     center4, range4)  # off E4
    Ray_TP3 = IntPolT2(Ray_E4, coeffpolar, originpolar3, p3)  # P3
    Ray_E5 = ReflEll(Ray_TP3, thet5, origin5, coeffellipse56,
                     center5, range5)  # OFF E5
    Ray_TP4 = IntPolT2(Ray_E5, coeffpolar, originpolar4, p4)
    Ray_E72 = ReflEll(Ray_TP4, thet7, origin7, coeffellipse7,
                      center7, range7)  # OFF E7
    return Ray_E72


def RRRRioM_first_half(Ri, p1, p2, p3, p4):
    Ray1 = ReflEll(Ri, thet10, origin10, coeffellipse7, center10, range10)
    Ray_RP1 = IntPolR2(Ray1, coeffpolar, originpolar1, p1)  # p1
    Ray_E9 = ReflEll(Ray_RP1, thet5, origin9,
                     coeffellipse56, center9, range9)  # E9
    Ray_RP2 = IntPolR2(Ray_E9, coeffpolar, originpolar2, p2)  # P2
    Ray_E3 = ReflEll(Ray_RP2, thet, origin3,
                     coeffellipse, center3, range3)  # E3
    return Ray_E3


# @jit(cache=True)
def RRRRioM_second_half(Ri, p1, p2, p3, p4, originM):
    Ray_M0 = IntM2(Ri, coeffmirr, originM)  # off mirror
    Ray_E4 = ReflEll(Ray_M0, thet, origin4, coeffellipse,
                     center4, range4)  # off E4
    Ray_RP3 = IntPolR2(Ray_E4, coeffpolar, originpolar3, p3)  # P3
    Ray_E6 = ReflEll(Ray_RP3, thet6, origin6, coeffellipse56,
                     center6, range6)  # off E6
    Ray_RP4 = IntPolR2(Ray_E6, coeffpolar, originpolar4, p4)
    Ray_E72 = ReflEll(Ray_RP4, thet7, origin7, coeffellipse7, center7, range7)
    return Ray_E72


def TTRRioM_first_half(Ri, p1, p2, p3, p4):
    Ray1 = ReflEll(Ri, thet10, origin10, coeffellipse7, center10, range10)
    Ray_TP1 = IntPolT2(Ray1, coeffpolar, originpolar1, p1)  # P1
    Ray_E8 = ReflEll(Ray_TP1, thet6, origin8,
                     coeffellipse56, center8, range8)  # E8
    Ray_TP2 = IntPolT2(Ray_E8, coeffpolar, originpolar2, p2)  # P2
    Ray_E3 = ReflEll(Ray_TP2, thet, origin3,
                     coeffellipse, center3, range3)  # E3
    return Ray_E3


# @jit(cache=True)
def TTRRioM_second_half(Ri, p1, p2, p3, p4, originM):
    Ray_M0 = IntM2(Ri, coeffmirr, originM)  # off mirror
    Ray_E4 = ReflEll(Ray_M0, thet, origin4, coeffellipse,
                     center4, range4)  # off E4
    Ray_RP3 = IntPolR2(Ray_E4, coeffpolar, originpolar3, p3)  # P3
    Ray_E6 = ReflEll(Ray_RP3, thet6, origin6, coeffellipse56,
                     center6, range6)  # off E6
    Ray_RP4 = IntPolR2(Ray_E6, coeffpolar, originpolar4, p4)
    Ray_E72 = ReflEll(Ray_RP4, thet7, origin7, coeffellipse7, center7, range7)
    return Ray_E72


def RTTRioM_first_half(Ri, p1, p2, p3, p4):
    Ray1 = ReflEll(Ri, thet10, origin10, coeffellipse7, center10, range10)
    Ray_RP1 = IntPolR2(Ray1, coeffpolar, originpolar1, p1)  # P1
    Ray_E9 = ReflEll(Ray_RP1, thet5, origin9,
                     coeffellipse56, center9, range9)  # E9
    Ray_TP2 = IntPolT2(Ray_E9, coeffpolar, originpolar2, p2)  # P2
    Ray_E1 = ReflEll(Ray_TP2, thet, origin1,
                     coeffellipse, center1, range1)  # E1
    return Ray_E1


# @jit(cache=True)
def RTTRioM_second_half(Ri, p1, p2, p3, p4, originM):
    Ray_M0 = IntM2(Ri, coeffmirr, originM)  # off mirror
    Ray_E2 = ReflEll(Ray_M0, thet, origin2, coeffellipse,
                     center2, range2)  # off E2
    Ray_TP3 = IntPolT2(Ray_E2, coeffpolar, originpolar3, p3)  # P3
    Ray_E6 = ReflEll(Ray_TP3, thet6, origin6,
                     coeffellipse56, center6, range6)  # E6
    Ray_RP4 = IntPolR2(Ray_E6, coeffpolar, originpolar4, p4)
    Ray_E72 = ReflEll(Ray_RP4, thet7, origin7, coeffellipse7, center7, range7)
    return Ray_E72


def RTRTioM_first_half(Ri, p1, p2, p3, p4):
    Ray1 = ReflEll(Ri, thet10, origin10, coeffellipse7, center10, range10)
    Ray_RP1 = IntPolR2(Ray1, coeffpolar, originpolar1, p1)  # P1
    Ray_E9 = ReflEll(Ray_RP1, thet5, origin9,
                     coeffellipse56, center9, range9)  # E9
    Ray_TP2 = IntPolT2(Ray_E9, coeffpolar, originpolar2, p2)  # P2
    Ray_E1 = ReflEll(Ray_TP2, thet, origin1,
                     coeffellipse, center1, range1)  # E1
    return Ray_E1


# @ jit(cache=True)
def RTRTioM_second_half(Ri, p1, p2, p3, p4, originM):
    Ray_M0 = IntM2(Ri, coeffmirr, originM)  # off mirror
    Ray_E2 = ReflEll(Ray_M0, thet, origin2, coeffellipse,
                     center2, range2)  # off E2
    Ray_RP3 = IntPolR2(Ray_E2, coeffpolar, originpolar3, p3)  # P3
    Ray_E5 = ReflEll(Ray_RP3, thet5, origin5,
                     coeffellipse56, center5, range5)  # E5
    Ray_TP4 = IntPolT2(Ray_E5, coeffpolar, originpolar4, p4)
    Ray_E72 = ReflEll(Ray_TP4, thet7, origin7, coeffellipse7, center7, range7)
    return Ray_E72


def TRRTioM_first_half(Ri, p1, p2, p3, p4):
    Ray1 = ReflEll(Ri, thet10, origin10, coeffellipse7, center10, range10)
    Ray_TP1 = IntPolT2(Ray1, coeffpolar, originpolar1, p1)  # P1
    Ray_E8 = ReflEll(Ray_TP1, thet6, origin8,
                     coeffellipse56, center8, range8)  # E8
    Ray_RP2 = IntPolR2(Ray_E8, coeffpolar, originpolar2, p2)  # P2
    Ray_E1 = ReflEll(Ray_RP2, thet, origin1,
                     coeffellipse, center1, range1)  # E3
    return Ray_E1


# @ jit(cache=True)
def TRRTioM_second_half(Ri, p1, p2, p3, p4, originM):
    Ray_M0 = IntM2(Ri, coeffmirr, originM)  # off mirror
    Ray_E2 = ReflEll(Ray_M0, thet, origin2, coeffellipse,
                     center2, range2)  # off E4
    Ray_RP3 = IntPolR2(Ray_E2, coeffpolar, originpolar3, p3)  # P3
    Ray_E5 = ReflEll(Ray_RP3, thet5, origin5, coeffellipse56, center5, range5)
    Ray_TP4 = IntPolT2(Ray_E5, coeffpolar, originpolar4, p4)
    Ray_E72 = ReflEll(Ray_TP4, thet7, origin7, coeffellipse7, center7, range7)
    return Ray_E72


def RRTTioM_first_half(Ri, p1, p2, p3, p4):
    Ray1 = ReflEll(Ri, thet10, origin10, coeffellipse7, center10, range10)
    Ray_RP1 = IntPolR2(Ray1, coeffpolar, originpolar1, p1)  # p1
    Ray_E9 = ReflEll(Ray_RP1, thet5, origin9,
                     coeffellipse56, center9, range9)  # E8
    Ray_RP2 = IntPolR2(Ray_E9, coeffpolar, originpolar2, p2)  # P2
    Ray_E3 = ReflEll(Ray_RP2, thet, origin3,
                     coeffellipse, center3, range3)  # E3
    return Ray_E3


# @ jit(cache=True)
def RRTTioM_second_half(Ri, p1, p2, p3, p4, originM):
    Ray_M0 = IntM2(Ri, coeffmirr, originM)  # off mirror
    Ray_E4 = ReflEll(Ray_M0, thet, origin4, coeffellipse,
                     center4, range4)  # off E4
    Ray_TP3 = IntPolT2(Ray_E4, coeffpolar, originpolar3, p3)  # P3
    Ray_E5 = ReflEll(Ray_TP3, thet5, origin5, coeffellipse56, center5, range5)
    Ray_TP4 = IntPolT2(Ray_E5, coeffpolar, originpolar4, p4)
    Ray_E72 = ReflEll(Ray_TP4, thet7, origin7, coeffellipse7, center7, range7)
    return Ray_E72


def TRTRioM_first_half(Ri, p1, p2, p3, p4):
    Ray1 = ReflEll(Ri, thet10, origin10, coeffellipse7, center10, range10)
    Ray_TP1 = IntPolT2(Ray1, coeffpolar, originpolar1, p1)  # P1
    Ray_E8 = ReflEll(Ray_TP1, thet6, origin8,
                     coeffellipse56, center8, range8)  # E8
    Ray_RP2 = IntPolR2(Ray_E8, coeffpolar, originpolar2, p2)  # P2
    Ray_E1 = ReflEll(Ray_RP2, thet, origin1,
                     coeffellipse, center1, range1)  # E3
    return Ray_E1


# @ jit(cache=True)
def TRTRioM_second_half(Ri, p1, p2, p3, p4, originM):
    Ray_M0 = IntM2(Ri, coeffmirr, originM)  # off mirror
    Ray_E2 = ReflEll(Ray_M0, thet, origin2, coeffellipse,
                     center2, range2)  # off E4
    Ray_TP3 = IntPolT2(Ray_E2, coeffpolar, originpolar3, p3)  # P3
    Ray_E6 = ReflEll(Ray_TP3, thet6, origin6, coeffellipse56, center6, range6)
    Ray_RP4 = IntPolR2(Ray_E6, coeffpolar, originpolar4, p4)
    Ray_E72 = ReflEll(Ray_RP4, thet7, origin7, coeffellipse7, center7, range7)
    return Ray_E72


'''the functions below are identical to above, except are 'To Pickle' such that
it returns every single ray created at every single step. All those below
require the position of the mirror (as an origin)'''


def TTTTioMPickle(Ri, p1, p2, p3, p4, originM):
    Ray1 = ReflEll(Ri, thet10, origin10, coeffellipse7, center10, range10)
    Ray_TP1 = IntPolT2(Ray1, coeffpolar, originpolar1, p1)  # P1
    Ray_E8 = ReflEll(Ray_TP1, thet6, origin8,
                     coeffellipse56, center8, range8)  # E8
    Ray_TP2 = IntPolT2(Ray_E8, coeffpolar, originpolar2, p2)  # P2
    Ray_E3 = ReflEll(Ray_TP2, thet, origin3,
                     coeffellipse, center3, range3)  # E3
    Ray_M0 = IntM2(Ray_E3, coeffmirr, originM)  # off mirror
    Ray_E4 = ReflEll(Ray_M0, thet, origin4, coeffellipse,
                     center4, range4)  # off E4
    Ray_TP3 = IntPolT2(Ray_E4, coeffpolar, originpolar3, p3)  # P3
    Ray_E5 = ReflEll(Ray_TP3, thet5, origin5, coeffellipse56,
                     center5, range5)  # OFF E5
    Ray_TP4 = IntPolT2(Ray_E5, coeffpolar, originpolar4, p4)
    Ray_E72 = ReflEll(Ray_TP4, thet7, origin7, coeffellipse7, center7, range7)
    return Ri, Ray1, Ray_TP1, Ray_E8, Ray_TP2, Ray_E3, Ray_M0, Ray_E4, Ray_TP3, Ray_E5, Ray_TP4, Ray_E72


def RRRRioMPickle(Ri, p1, p2, p3, p4, originM):
    Ray1 = ReflEll(Ri, thet10, origin10, coeffellipse7, center10, range10)
    Ray_RP1 = IntPolR2(Ray1, coeffpolar, originpolar1, p1)  # p1
    Ray_E9 = ReflEll(Ray_RP1, thet5, origin9,
                     coeffellipse56, center9, range9)  # E9
    Ray_RP2 = IntPolR2(Ray_E9, coeffpolar, originpolar2, p2)  # P2
    Ray_E3 = ReflEll(Ray_RP2, thet, origin3,
                     coeffellipse, center3, range3)  # E3
    Ray_M0 = IntM2(Ray_E3, coeffmirr, originM)  # off mirror
    Ray_E4 = ReflEll(Ray_M0, thet, origin4, coeffellipse,
                     center4, range4)  # off E4
    Ray_RP3 = IntPolR2(Ray_E4, coeffpolar, originpolar3, p3)  # P3
    Ray_E6 = ReflEll(Ray_RP3, thet6, origin6, coeffellipse56,
                     center6, range6)  # off E6
    Ray_RP4 = IntPolR2(Ray_E6, coeffpolar, originpolar4, p4)
    Ray_E72 = ReflEll(Ray_RP4, thet7, origin7, coeffellipse7, center7, range7)
    return Ri, Ray1, Ray_RP1, Ray_E9, Ray_RP2, Ray_E3, Ray_M0, Ray_E4, Ray_RP3, Ray_E6, Ray_RP4, Ray_E72


def TTRRioMPickle(Ri, p1, p2, p3, p4, originM):
    Ray1 = ReflEll(Ri, thet10, origin10, coeffellipse7, center10, range10)
    Ray_TP1 = IntPolT2(Ray1, coeffpolar, originpolar1, p1)  # P1
    Ray_E8 = ReflEll(Ray_TP1, thet6, origin8,
                     coeffellipse56, center8, range8)  # E8
    Ray_TP2 = IntPolT2(Ray_E8, coeffpolar, originpolar2, p2)  # P2
    Ray_E3 = ReflEll(Ray_TP2, thet, origin3,
                     coeffellipse, center3, range3)  # E3
    Ray_M0 = IntM2(Ray_E3, coeffmirr, originM)  # off mirror
    Ray_E4 = ReflEll(Ray_M0, thet, origin4, coeffellipse,
                     center4, range4)  # off E4
    Ray_RP3 = IntPolR2(Ray_E4, coeffpolar, originpolar3, p3)  # P3
    Ray_E6 = ReflEll(Ray_RP3, thet6, origin6, coeffellipse56,
                     center6, range6)  # off E6
    Ray_RP4 = IntPolR2(Ray_E6, coeffpolar, originpolar4, p4)
    Ray_E72 = ReflEll(Ray_RP4, thet7, origin7, coeffellipse7, center7, range7)
    return Ri, Ray1, Ray_TP1, Ray_E8, Ray_TP2, Ray_E3, Ray_M0, Ray_E4, Ray_RP3, Ray_E6, Ray_RP4, Ray_E72


def RTTRioMPickle(Ri, p1, p2, p3, p4, originM):
    Ray1 = ReflEll(Ri, thet10, origin10, coeffellipse7, center10, range10)
    Ray_RP1 = IntPolR2(Ray1, coeffpolar, originpolar1, p1)  # P1
    Ray_E9 = ReflEll(Ray_RP1, thet5, origin9,
                     coeffellipse56, center9, range9)  # E9
    Ray_TP2 = IntPolT2(Ray_E9, coeffpolar, originpolar2, p2)  # P2
    Ray_E1 = ReflEll(Ray_TP2, thet, origin1,
                     coeffellipse, center1, range1)  # E1
    Ray_M0 = IntM2(Ray_E1, coeffmirr, originM)  # off mirror
    Ray_E2 = ReflEll(Ray_M0, thet, origin2, coeffellipse,
                     center2, range2)  # off E2
    Ray_TP3 = IntPolT2(Ray_E2, coeffpolar, originpolar3, p3)  # P3
    Ray_E6 = ReflEll(Ray_TP3, thet6, origin6,
                     coeffellipse56, center6, range6)  # E6
    Ray_RP4 = IntPolR2(Ray_E6, coeffpolar, originpolar4, p4)
    Ray_E72 = ReflEll(Ray_RP4, thet7, origin7, coeffellipse7, center7, range7)
    return Ri, Ray1, Ray_RP1, Ray_E9, Ray_TP2, Ray_E1, Ray_M0, Ray_E2, Ray_TP3, Ray_E6, Ray_RP4, Ray_E72


def RTRTioMPickle(Ri, p1, p2, p3, p4, originM):
    Ray1 = ReflEll(Ri, thet10, origin10, coeffellipse7, center10, range10)
    Ray_RP1 = IntPolR2(Ray1, coeffpolar, originpolar1, p1)  # P1
    Ray_E9 = ReflEll(Ray_RP1, thet5, origin9,
                     coeffellipse56, center9, range9)  # E9
    Ray_TP2 = IntPolT2(Ray_E9, coeffpolar, originpolar2, p2)  # P2
    Ray_E1 = ReflEll(Ray_TP2, thet, origin1,
                     coeffellipse, center1, range1)  # E1
    Ray_M0 = IntM2(Ray_E1, coeffmirr, originM)  # off mirror
    Ray_E2 = ReflEll(Ray_M0, thet, origin2, coeffellipse,
                     center2, range2)  # off E2
    Ray_RP3 = IntPolR2(Ray_E2, coeffpolar, originpolar3, p3)  # P3
    Ray_E5 = ReflEll(Ray_RP3, thet5, origin5,
                     coeffellipse56, center5, range5)  # E5
    Ray_TP4 = IntPolT2(Ray_E5, coeffpolar, originpolar4, p4)
    Ray_E72 = ReflEll(Ray_TP4, thet7, origin7, coeffellipse7, center7, range7)
    return Ri, Ray1, Ray_RP1, Ray_E9, Ray_TP2, Ray_E1, Ray_M0, Ray_E2, Ray_RP3, Ray_E5, Ray_TP4, Ray_E72


def TRRTioMPickle(Ri, p1, p2, p3, p4, originM):
    Ray1 = ReflEll(Ri, thet10, origin10, coeffellipse7, center10, range10)
    Ray_TP1 = IntPolT2(Ray1, coeffpolar, originpolar1, p1)  # P1
    Ray_E8 = ReflEll(Ray_TP1, thet6, origin8,
                     coeffellipse56, center8, range8)  # E8
    Ray_RP2 = IntPolR2(Ray_E8, coeffpolar, originpolar2, p2)  # P2
    Ray_E1 = ReflEll(Ray_RP2, thet, origin1,
                     coeffellipse, center1, range1)  # E3
    Ray_M0 = IntM2(Ray_E1, coeffmirr, originM)  # off mirror
    Ray_E2 = ReflEll(Ray_M0, thet, origin2, coeffellipse,
                     center2, range2)  # off E4
    Ray_RP3 = IntPolR2(Ray_E2, coeffpolar, originpolar3, p3)  # P3
    Ray_E5 = ReflEll(Ray_RP3, thet5, origin5, coeffellipse56, center5, range5)
    Ray_TP4 = IntPolT2(Ray_E5, coeffpolar, originpolar4, p4)
    Ray_E72 = ReflEll(Ray_TP4, thet7, origin7, coeffellipse7, center7, range7)
    return Ri, Ray1, Ray_TP1, Ray_E8, Ray_RP2, Ray_E1, Ray_M0, Ray_E2, Ray_RP3, Ray_E5, Ray_TP4, Ray_E72


def RRTTioMPickle(Ri, p1, p2, p3, p4, originM):
    Ray1 = ReflEll(Ri, thet10, origin10, coeffellipse7, center10, range10)
    Ray_RP1 = IntPolR2(Ray1, coeffpolar, originpolar1, p1)  # p1
    Ray_E9 = ReflEll(Ray_RP1, thet5, origin9,
                     coeffellipse56, center9, range9)  # E8
    Ray_RP2 = IntPolR2(Ray_E9, coeffpolar, originpolar2, p2)  # P2
    Ray_E3 = ReflEll(Ray_RP2, thet, origin3,
                     coeffellipse, center3, range3)  # E3
    Ray_M0 = IntM2(Ray_E3, coeffmirr, originM)  # off mirror
    Ray_E4 = ReflEll(Ray_M0, thet, origin4, coeffellipse,
                     center4, range4)  # off E4
    Ray_TP3 = IntPolT2(Ray_E4, coeffpolar, originpolar3, p3)  # P3
    Ray_E5 = ReflEll(Ray_TP3, thet5, origin5, coeffellipse56, center5, range5)
    Ray_TP4 = IntPolT2(Ray_E5, coeffpolar, originpolar4, p4)
    Ray_E72 = ReflEll(Ray_TP4, thet7, origin7, coeffellipse7, center7, range7)
    return Ri, Ray1, Ray_RP1, Ray_E9, Ray_RP2, Ray_E3, Ray_M0, Ray_E4, Ray_TP3, Ray_E5, Ray_TP4, Ray_E72


def TRTRioMPickle(Ri, p1, p2, p3, p4, originM):
    Ray1 = ReflEll(Ri, thet10, origin10, coeffellipse7, center10, range10)
    Ray_TP1 = IntPolT2(Ray1, coeffpolar, originpolar1, p1)  # P1
    Ray_E8 = ReflEll(Ray_TP1, thet6, origin8,
                     coeffellipse56, center8, range8)  # E8
    Ray_RP2 = IntPolR2(Ray_E8, coeffpolar, originpolar2, p2)  # P2
    Ray_E1 = ReflEll(Ray_RP2, thet, origin1,
                     coeffellipse, center1, range1)  # E3
    Ray_M0 = IntM2(Ray_E1, coeffmirr, originM)  # off mirror
    Ray_E2 = ReflEll(Ray_M0, thet, origin2, coeffellipse,
                     center2, range2)  # off E4
    Ray_TP3 = IntPolT2(Ray_E2, coeffpolar, originpolar3, p3)  # P3
    Ray_E6 = ReflEll(Ray_TP3, thet6, origin6, coeffellipse56, center6, range6)
    Ray_RP4 = IntPolR2(Ray_E6, coeffpolar, originpolar4, p4)
    Ray_E72 = ReflEll(Ray_RP4, thet7, origin7, coeffellipse7, center7, range7)
    return Ri, Ray1, Ray_TP1, Ray_E8, Ray_RP2, Ray_E1, Ray_M0, Ray_E2, Ray_TP3, Ray_E6, Ray_RP4, Ray_E72


# Ignore Below
'''Tilting: (includes ability to tilt mirror and polarizers)
GIVE initial RAYS AND Y position, returns output rays from detector if mirror at Y. Includes tilt of polarizers and tilt of mirror.
Not completed.

def OFDMtilt(Rays,y,thetmirr,thetpolar):
    Rayf = []
    for i in range(len(Rays)):
        Paths = [TTTTioMtilt,RRRRioMtilt,TTRRioMtilt,RTTRioMtilt,
            RTRTioMtilt,TRRTioMtilt,RRTTioMtilt,TRTRioMtilt]
        Ri = Rays[i]
        for j in range(8):
            origin = (0,y,0)
            out = Paths[j](Ri,p1,p2,p3,p4,origin,thetmirr,thetpolar)
            if out is not None:
                Rayf.append(out)
    return Rayf


give rays to be run through this FTS at a specific y, returns the good rays and the region. not used anymore
# just give number of rays to be run through this FTS at a specific y!
def RunRaysMtilt(Rays,y,thetmirr,thetpolar):
    Rayf = OFDMtilt(Rays,y,thetmirr,thetpolar)
    G= checkoutraysM(Rayf,[160.375,-113],7.9375) # GOOD RAYS ONLY
    Gtestsorted = sortgrid(G)
    return Gtestsorted

def makeraysTILT(sourcepointorigin,r,n,):
    sourcethet = [0.,0.,0.] #SHOT STRAIGHT UP
    sourcepoints = specsource(r,sourcepointorigin,sourcethet,n) # SOURCE
    Rays = FSRay(n,sourcepoints, sourcethet,origin10)
    for i in range(n):
        v1x,v1y,v1z = transformLG(v[0],v[1],v[2],originG,sourcethet)
        v2 = sepop(v1x,v1y,v1z)
        Rays[i][3]=v2
    return Rays

def makeraysTILT_Zero(sourcepointorigin,r,n,):
    sourcethet = [0.,0.,0.] #SHOT STRAIGHT UP
    sourcepoints = specsource(r,sourcepointorigin,sourcethet,n) # SOURCE
    Rays = FSRay(n,sourcepoints, sourcethet,origin10)
    for i in range(n):
        v1x,v1y,v1z = transformLG(v[0],v[1],v[2],originG,sourcethet)
        v2 = sepop(v1x,v1y,v1z)
        Rays[i][3]=v2
    return Rays


def IntM2tilt(Ray,thetmirr,coeffmirr,originmirr):
    if Ray is None:
        return
    p = Ray[2]
    v = Ray[3]
    Ray_M = []
    Ray_M.append(Ray[0])
    Ray_M.append(Ray[1])
    if (np.abs(thetmirr[0]) + np.abs(thetmirr[1]) + np.abs(thetmirr[2])) == 0.0:
        return IntM2(Ray,coeffmirr,originmirr)
    else:
        p1,v1 = RT(p,v,thetG,originG,thetmirr,originmirr) #LOCAL
        intpoint = PLINTyS(originmirr[1],p1,v1)
        if SRM(intpoint,coeffmirr,originmirr) == True:
            VectLNorm = N(v1) #local
            PNormpos = [0,-1,0] #from definition of mirror (check sign what)
            PNormneg = [0,1,0] #what
            VectReflectpos = VectLNorm -2* \
                N(PNormpos)*(np.dot(VectLNorm,N(PNormpos))) #Local
            VectReflectneg = VectLNorm -2* \
                N(PNormneg)*(np.dot(VectLNorm,N(PNormneg)))
            GoodInt,GoodVect = finddirec(
                p1,v1,intpoint,intpoint,VectReflectpos,VectReflectneg)
            p2,v2 = RT(GoodInt,GoodVect,thetmirr,
                       originmirr,thetG,originG) #NOW GLOBAL
            Ray_M.append(p2)
            Ray_M.append(v2) #change to reflected
            Df = dist(p,GoodInt) #ADDING DISTANCE TRAVELLED (ADDED 4/21)
            Ray_M.append(Ray[4]+Df)
            return Ray_M
        else:
            return

def IntPolR2Tilt(Ray,coeffpolar,originpolar,PolarizerAngle,thetpolar): #reflected
    if Ray is None: #just bug check
        return
    if (np.abs(thetpolar[0]) + np.abs(thetpolar[1]) + np.abs(thetpolar[2])) == 0.0:
        return IntPolR2(Ray,coeffpolar,originpolar,PolarizerAngle)
    else:
        thet_polarized = Ray[0] #theta
        I = Ray[1] #intensity
        p = Ray[2] #point
        v = Ray[3] #vector
        Di = Ray[4] #distance
        Ex,Ey = np.sqrt(I)*np.cos(thet_polarized),np.sqrt(I)* \
                        np.sin(thet_polarized)
        Ray_R = [] #thet, I, intpoint, vects
        Ray_R.append(PolarizerAngle) #same vector of course
        I_R = PolarizerInteractionR(Ex,Ey,thet_polarized,PolarizerAngle)
        Ray_R.append(I_R)
        p1,v1 = RT(p,v,thetG,originG,thetpolar,originpolar) #LOCAL
        intpoint = PLINTyS(originpolar[1],p1,v1)
        if SRM(intpoint,coeffpolar,originpolar) == True:
            VectLNorm = N(v1)
            PNormpos = [0,-1,0]
            PNormneg = [0,1,0]
            VectReflectpos = VectLNorm -2* \
                N(PNormpos)*(np.dot(VectLNorm,N(PNormpos))) #Local
            VectReflectneg = VectLNorm -2* \
                N(PNormneg)*(np.dot(VectLNorm,N(PNormneg)))
            GoodInt,GoodVect = finddirec(
                p1,v1,intpoint,intpoint,VectReflectpos,VectReflectneg)
            p2,v2 = RT(GoodInt,GoodVect,thetpolar,
                       originpolar,thetG,originG) #NOW GLOBAL
            Ray_R.append(p2)
            Ray_R.append(v2) #change to reflected
            Df = dist(p,GoodInt)
            Ray_R.append(Di + Df)
        else:
            return
        return Ray_R

def RunOneRayTilt(Lamd,Nsize,spo,tilt):
    n = 1
    r = 0
    Rays = makeraysTILT(spo,r,n,v)
    jx,jy = gridlines(7.9375,[160.375,-113],200) #these are now the PIXELS
    Pix = MakePixels(jx,jy) #center of each pixel
    Ij = []
    Delay = []
    for y in np.linspace(-18,18,int(Nsize)):
        PTot=0
        OutRays=RunRaysM(Rays,y) #eight each
        # Overlap = gaussoverlap(OutRays[0],OutRays[5],3.3) #two paths that hit two different spots
        for j in range(len(Pix)): #per PIXEL
            Ex4i = 0 #adding PER PIXEL from parts of RAYS in this PIXEL
            Ey4i = 0 #THIS IS WHERE THEY WILL INTERFERE
            for i in range(len(OutRays)): #per ray IN THIS PIXEL
                I = OutRays[i][1]
                thet = OutRays[i][0]
                phase = np.exp(1j*(OutRays[i][4]*2*np.pi/Lamd)) #factor of 2??
                Ex1 = np.sqrt(I)*np.cos(thet)
                Ey1 = np.sqrt(I)*np.sin(thet)
                Ex = Ex1*phase
                Ey = Ey1*phase
                # doing summation over entire detector
                sig,mux,muy = MakeGaussian(OutRays[i],Lamd)
                Gr = Airygaussian3dNORM(Pix[j][0],Pix[j][1],sig,mux,muy)
                Ex4i = Ex4i + Gr*Ex
                Ey4i = Ey4i + Gr*Ey
            PTot = PTot + (Ex4i*Ex4i.conjugate()).real + \
                           (Ey4i*Ey4i.conjugate()).real
        Delay.append(y*0.95630475596*4)
        Ij.append(PTot)
    return Delay,Ij

def RunOneRayTiltSource(Lamd,Nsize,spo,thetmirr,thetpolar):
    n = 1
    r = 0
    Rays = makeraysVERTICAL(spo,r,n)
    jx,jy = gridlines(7.9375,[160.375,-113],200) #these are now the PIXELS
    Pix = MakePixels(jx,jy) #center of each pixel
    Ij = []
    Delay = []
    for y in np.linspace(-18,18,int(Nsize)):
        PTot=0
        OutRays=RunRaysMtilt(Rays,y,thetmirr,thetpolar) #eight each
        for j in range(len(Pix)): #per PIXEL
            Ex4i = 0 #adding PER PIXEL from parts of RAYS in this PIXEL
            Ey4i = 0 #THIS IS WHERE THEY WILL INTERFERE
            for i in range(len(OutRays)): #per ray IN THIS PIXEL
                I = OutRays[i][1]
                thet = OutRays[i][0]
                phase = np.exp(1j*(OutRays[i][4]*2*np.pi/Lamd)) #factor of 2??
                Ex1 = np.sqrt(I)*np.cos(thet)
                Ey1 = np.sqrt(I)*np.sin(thet)
                Ex = Ex1*phase
                Ey = Ey1*phase
                # doing summation over entire detector
                sig,mux,muy = MakeGaussian(OutRays[i],Lamd)
                Gr = Airygaussian3dNORM(Pix[j][0],Pix[j][1],sig,mux,muy)
                Ex4i = Ex4i + Gr*Ex
                Ey4i = Ey4i + Gr*Ey
            PTot = PTot + (Ex4i*Ex4i.conjugate()).real + \
                           (Ey4i*Ey4i.conjugate()).real
        Delay.append(y*0.95630475596*4)
        Ij.append(PTot)
    return Delay,Ij

# Making the source out of many different rays on a source
def RunSourceTiltVERT(Lamd,Nsize,spo,n,r,thetmirr,thetpolar):
    Rays = makeraysVERTICAL(spo,r,n) #sourcethet as [0,0,0]
    jx,jy = gridlines(7.9375,[160.375,-113],200) #these are now the PIXELS
    Pix = MakePixels(jx,jy) #center of each pixel
    Ij = []
    Delay = []
    for y in np.linspace(-18,18,int(Nsize)):
        PTot=0
        OutRays=RunRaysMtilt(Rays,y,thetmirr,thetpolar) #eight each
        for j in range(len(Pix)): #per PIXEL
            Ex4i = 0 #adding PER PIXEL from parts of RAYS in this PIXEL
            Ey4i = 0 #THIS IS WHERE THEY WILL INTERFERE
            for i in range(len(OutRays)): #per ray IN THIS PIXEL
                I = OutRays[i][1]
                thet = OutRays[i][0]
                phase = np.exp(1j*(OutRays[i][4]*2*np.pi/Lamd)) #factor of 2??
                Ex1 = np.sqrt(I)*np.cos(thet)
                Ey1 = np.sqrt(I)*np.sin(thet)
                Ex = Ex1*phase
                Ey = Ey1*phase
                # doing summation over entire detector
                sig,mux,muy = MakeGaussian(OutRays[i],Lamd)
                Gr = Airygaussian3dNORM(Pix[j][0],Pix[j][1],sig,mux,muy)
                Ex4i = Ex4i + Gr*Ex
                Ey4i = Ey4i + Gr*Ey
            PTot = PTot + (Ex4i*Ex4i.conjugate()).real + \
                           (Ey4i*Ey4i.conjugate()).real
        Delay.append(y*0.95630475596*4)
        Ij.append(PTot)
    return Delay,Ij


def TTTTioMtilt(Ri,p1,p2,p3,p4,originM,thetmirr,thetpolar):
    Ray1 = ReflEll(Ri,thet10,origin10,coeffellipse7,center10,range10)
    Ray_TP1 = IntPolT2(Ray1,coeffpolar,originpolar1,p1) #P1
    Ray_E8 = ReflEll(Ray_TP1,thet6,origin8,coeffellipse56,center8,range8) #E8
    Ray_TP2 = IntPolT2(Ray_E8,coeffpolar,originpolar2,p2) #P2
    Ray_E3 = ReflEll(Ray_TP2,thet,origin3,coeffellipse,center3,range3) #E3
    Ray_M0 = IntM2tilt(Ray_E3, thetmirr,coeffmirr, originM) #off mirror
    Ray_E4 = ReflEll(Ray_M0, thet,origin4,coeffellipse,center4,range4) #off E4
    Ray_TP3 = IntPolT2(Ray_E4,coeffpolar,originpolar3,p3) #P3
    Ray_E5 = ReflEll(Ray_TP3, thet5,origin5,
                     coeffellipse56,center5,range5)#OFF E5
    Ray_TP4 = IntPolT2(Ray_E5,coeffpolar,originpolar4,p4)
    Ray_E72 = ReflEll(Ray_TP4,thet7,origin7,coeffellipse7,center7,range7)
    return Ray_E72

def RRRRioMtilt(Ri,p1,p2,p3,p4,originM,thetmirr,thetpolar):
    Ray1 = ReflEll(Ri,thet10,origin10,coeffellipse7,center10,range10)
    Ray_RP1 = IntPolR2Tilt(Ray1,coeffpolar,originpolar1,p1,thetpolar[0])#p1
    Ray_E9 = ReflEll(Ray_RP1,thet5,origin9,coeffellipse56,center9,range9) #E9
    Ray_RP2 = IntPolR2Tilt(Ray_E9,coeffpolar,originpolar2,p2,thetpolar[1]) #P2
    Ray_E3 = ReflEll(Ray_RP2,thet,origin3,coeffellipse,center3,range3) #E3
    Ray_M0 = IntM2tilt(Ray_E3,thetmirr, coeffmirr, originM) #off mirror
    Ray_E4 = ReflEll(Ray_M0, thet,origin4,coeffellipse,center4,range4) #off E4
    Ray_RP3 = IntPolR2Tilt(Ray_E4,coeffpolar,originpolar3,p3,thetpolar[2]) #P3
    Ray_E6 = ReflEll(Ray_RP3, thet6,origin6,
                     coeffellipse56,center6,range6) #off E6
    Ray_RP4 = IntPolR2Tilt(Ray_E6,coeffpolar,originpolar4,p4,thetpolar[3])
    Ray_E72 = ReflEll(Ray_RP4,thet7,origin7,coeffellipse7,center7,range7)
    return Ray_E72

def TTRRioMtilt(Ri,p1,p2,p3,p4,originM,thetmirr,thetpolar):
    Ray1 = ReflEll(Ri,thet10,origin10,coeffellipse7,center10,range10)
    Ray_TP1 = IntPolT2(Ray1,coeffpolar,originpolar1,p1) #P1
    Ray_E8 = ReflEll(Ray_TP1,thet6,origin8,coeffellipse56,center8,range8) #E8
    Ray_TP2 = IntPolT2(Ray_E8,coeffpolar,originpolar2,p2) #P2
    Ray_E3 = ReflEll(Ray_TP2,thet,origin3,coeffellipse,center3,range3) #E3
    Ray_M0 = IntM2tilt(Ray_E3,thetmirr, coeffmirr, originM) #off mirror
    Ray_E4 = ReflEll(Ray_M0, thet,origin4,coeffellipse,center4,range4) #off E4
    Ray_RP3 = IntPolR2Tilt(Ray_E4,coeffpolar,originpolar3,p3,thetpolar[2]) #P3
    Ray_E6 = ReflEll(Ray_RP3, thet6,origin6,
                     coeffellipse56,center6,range6) #off E6
    Ray_RP4 = IntPolR2Tilt(Ray_E6,coeffpolar,originpolar4,p4,thetpolar[3])
    Ray_E72 = ReflEll(Ray_RP4,thet7,origin7,coeffellipse7,center7,range7)
    return Ray_E72

def RTTRioMtilt(Ri,p1,p2,p3,p4,originM,thetmirr,thetpolar):
    Ray1 = ReflEll(Ri,thet10,origin10,coeffellipse7,center10,range10)
    Ray_RP1 = IntPolR2Tilt(Ray1,coeffpolar,originpolar1,p1,thetpolar[0]) #P1
    Ray_E9 = ReflEll(Ray_RP1,thet5,origin9,coeffellipse56,center9,range9) #E9
    Ray_TP2 = IntPolT2(Ray_E9,coeffpolar,originpolar2,p2) #P2
    Ray_E1 = ReflEll(Ray_TP2,thet,origin1,coeffellipse,center1,range1) #E1
    Ray_M0 = IntM2tilt(Ray_E1,thetmirr, coeffmirr, originM) #off mirror
    Ray_E2 = ReflEll(Ray_M0, thet,origin2,coeffellipse,center2,range2) #off E2
    Ray_TP3 = IntPolT2(Ray_E2,coeffpolar,originpolar3,p3) #P3
    Ray_E6 = ReflEll(Ray_TP3, thet6,origin6,coeffellipse56,center6,range6) #E6
    Ray_RP4 = IntPolR2Tilt(Ray_E6,coeffpolar,originpolar4,p4,thetpolar[3])
    Ray_E72 = ReflEll(Ray_RP4,thet7,origin7,coeffellipse7,center7,range7)
    return Ray_E72

def RTRTioMtilt(Ri,p1,p2,p3,p4,originM,thetmirr,thetpolar):
    Ray1 = ReflEll(Ri,thet10,origin10,coeffellipse7,center10,range10)
    Ray_RP1 = IntPolR2Tilt(Ray1,coeffpolar,originpolar1,p1,thetpolar[0]) #P1
    Ray_E9 = ReflEll(Ray_RP1,thet5,origin9,coeffellipse56,center9,range9) #E9
    Ray_TP2 = IntPolT2(Ray_E9,coeffpolar,originpolar2,p2) #P2
    Ray_E1 = ReflEll(Ray_TP2,thet,origin1,coeffellipse,center1,range1) #E1
    Ray_M0 = IntM2tilt(Ray_E1,thetmirr, coeffmirr, originM) #off mirror
    Ray_E2 = ReflEll(Ray_M0, thet,origin2,coeffellipse,center2,range2) #off E2
    Ray_RP3 = IntPolR2Tilt(Ray_E2,coeffpolar,originpolar3,p3,thetpolar[2]) #P3
    Ray_E5 = ReflEll(Ray_RP3, thet5,origin5,coeffellipse56,center5,range5) #E5
    Ray_TP4 = IntPolT2(Ray_E5,coeffpolar,originpolar4,p4)
    Ray_E72 = ReflEll(Ray_TP4,thet7,origin7,coeffellipse7,center7,range7)
    return Ray_E72

def TRRTioMtilt(Ri,p1,p2,p3,p4,originM,thetmirr,thetpolar):
    Ray1 = ReflEll(Ri,thet10,origin10,coeffellipse7,center10,range10)
    Ray_TP1 = IntPolT2(Ray1,coeffpolar,originpolar1,p1) #P1
    Ray_E8 = ReflEll(Ray_TP1,thet6,origin8,coeffellipse56,center8,range8) #E8
    Ray_RP2 = IntPolR2Tilt(Ray_E8,coeffpolar,originpolar2,p2,thetpolar[1]) #P2
    Ray_E1 = ReflEll(Ray_RP2,thet,origin1,coeffellipse,center1,range1) #E3
    Ray_M0 = IntM2tilt(Ray_E1,thetmirr, coeffmirr, originM) #off mirror
    Ray_E2 = ReflEll(Ray_M0, thet,origin2,coeffellipse,center2,range2) #off E4
    Ray_RP3 = IntPolR2Tilt(Ray_E2,coeffpolar,originpolar3,p3,thetpolar[2]) #P3
    Ray_E5 = ReflEll(Ray_RP3, thet5,origin5,coeffellipse56,center5,range5)
    Ray_TP4 = IntPolT2(Ray_E5,coeffpolar,originpolar4,p4)
    Ray_E72 = ReflEll(Ray_TP4,thet7,origin7,coeffellipse7,center7,range7)
    return Ray_E72

def RRTTioMtilt(Ri,p1,p2,p3,p4,originM,thetmirr,thetpolar):
    Ray1 = ReflEll(Ri,thet10,origin10,coeffellipse7,center10,range10)
    Ray_RP1 = IntPolR2Tilt(Ray1,coeffpolar,originpolar1,p1,thetpolar[0])#p1
    Ray_E9 = ReflEll(Ray_RP1,thet5,origin9,coeffellipse56,center9,range9) #E8
    Ray_RP2 = IntPolR2Tilt(Ray_E9,coeffpolar,originpolar2,p2,thetpolar[1]) #P2
    Ray_E3 = ReflEll(Ray_RP2,thet,origin3,coeffellipse,center3,range3) #E3
    Ray_M0 = IntM2tilt(Ray_E3,thetmirr, coeffmirr, originM) #off mirror
    Ray_E4 = ReflEll(Ray_M0, thet,origin4,coeffellipse,center4,range4) #off E4
    Ray_TP3 = IntPolT2(Ray_E4,coeffpolar,originpolar3,p3) #P3
    Ray_E5 = ReflEll(Ray_TP3, thet5,origin5,coeffellipse56,center5,range5)
    Ray_TP4 = IntPolT2(Ray_E5,coeffpolar,originpolar4,p4)
    Ray_E72 = ReflEll(Ray_TP4,thet7,origin7,coeffellipse7,center7,range7)
    return Ray_E72

def TRTRioMtilt(Ri,p1,p2,p3,p4,originM,thetmirr,thetpolar):
    Ray1 = ReflEll(Ri,thet10,origin10,coeffellipse7,center10,range10)
    Ray_TP1 = IntPolT2(Ray1,coeffpolar,originpolar1,p1) #P1
    Ray_E8 = ReflEll(Ray_TP1,thet6,origin8,coeffellipse56,center8,range8) #E8
    Ray_RP2 = IntPolR2Tilt(Ray_E8,coeffpolar,originpolar2,p2,thetpolar[1]) #P2
    Ray_E1 = ReflEll(Ray_RP2,thet,origin1,coeffellipse,center1,range1) #E3
    Ray_M0 = IntM2tilt(Ray_E1, thetmirr,coeffmirr, originM) #off mirror
    Ray_E2 = ReflEll(Ray_M0, thet,origin2,coeffellipse,center2,range2) #off E4
    Ray_TP3 = IntPolT2(Ray_E2,coeffpolar,originpolar3,p3) #P3
    Ray_E6 = ReflEll(Ray_TP3, thet6,origin6,coeffellipse56,center6,range6)
    Ray_RP4 = IntPolR2Tilt(Ray_E6,coeffpolar,originpolar4,p4,thetpolar[3])
    Ray_E72 = ReflEll(Ray_RP4,thet7,origin7,coeffellipse7,center7,range7)
    return Ray_E72

    '''


def reflect_line_about_plane(line_point, line_vec, plane_point, plane_vec):
    intersect_point = line_point - line_vec * (
        (line_point - plane_point).dot(plane_vec)) / (plane_vec.dot(line_vec))
    reflected_vector = line_vec - \
        (2 * (line_vec.dot(N(plane_vec))) * N(plane_vec))
    return intersect_point, reflected_vector


def reflect_dihedral_lower(incoming_point, incoming_vector,
                           dihedral_center_point):
    plane_normal = np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2)])
    intersection_point, reflected_vector = reflect_line_about_plane(
        incoming_point, incoming_vector, dihedral_center_point, plane_normal)
    return intersection_point, reflected_vector


def reflect_dihedral_upper(incoming_point, incoming_vector,
                           dihedral_center_point):
    plane_normal = np.array([0, 1 / np.sqrt(2), -1 / np.sqrt(2)])
    intersection_point, reflected_vector = reflect_line_about_plane(
        incoming_point, incoming_vector, dihedral_center_point, plane_normal)
    return intersection_point, reflected_vector


def reflect_dihedral_planes(incoming_point, incoming_vector,
                            dihedral_center_point):
    # Try reflecting off the lower mirror, then try reflecting off the upper...
    # Can really just find the planar intersection and then go that way

    # We're hitting the wrong plane here from the other side...
    center_intersect = PLINTyS(
        dihedral_center_point[1], incoming_point, incoming_vector)

    # If its z-value is positive, we reflect off the top of the mirror

    # this is only if we star on the positive side!
    # since if we're on the negative side the 'upper' and 'lower' mirrors
    # are switched
    if (center_intersect[2] > 0 and incoming_point[1] > 0) or (
            center_intersect[2] < 0 and incoming_point[1] < 0):
        funcs = [reflect_dihedral_upper, reflect_dihedral_lower]
    else:
        funcs = [reflect_dihedral_lower, reflect_dihedral_upper]

    intersect1, reflected_vector1 = funcs[0](incoming_point, incoming_vector,
                                             dihedral_center_point)
    intersect2, reflected_vector2 = funcs[1](intersect1, reflected_vector1,
                                             dihedral_center_point)
    return intersect1, reflected_vector1, intersect2, reflected_vector2


def raytrace_dihedral_mirror(incoming_ray, mirror_coefficients,
                             mirror_position):
    if incoming_ray is None:
        return None, None
    incoming_point = np.array(incoming_ray[2])
    incoming_vector = np.array(incoming_ray[3])
    # print(incoming_point, incoming_vector)
    intersect1, reflected_vector1, intersect2, reflected_vector2 =  \
        reflect_dihedral_planes(incoming_point, incoming_vector,
                                np.array(mirror_position))
    # print(intersect1, reflected_vector1, intersect2, reflected_vector2)
    first_reflected_ray = [incoming_ray[0], incoming_ray[1], list(intersect1),
                           list(reflected_vector1),
                           incoming_ray[4] + dist(incoming_point, intersect1)]
    # Make sure the x polarization flips while the y polarization does not!
    # So theta -> pi - theta
    internal_reflected_ray = [np.pi - incoming_ray[0], incoming_ray[1],
                              list(intersect2), list(reflected_vector2),
                              first_reflected_ray[4] + dist(intersect1,
                                                            intersect2)]
    return first_reflected_ray, internal_reflected_ray


def raytrace_other_mirror(incoming_ray, mirror_origin, mirror_normal_vec,
                          mirror_bounds):
    if incoming_ray is None:
        return None
    incoming_point = np.array(incoming_ray[2])
    incoming_vector = np.array(incoming_ray[3])
    # print(incoming_point, incoming_vector)

    # starting_normal = [0, 1, 0]
    # plane_normal = np.array([.82849, -.56001, 0])
    # plane_normal = rotate(starting_normal, mirror_tilt)
    plane_normal = np.array(mirror_normal_vec)
    intersect, reflected_vector = reflect_line_about_plane(
        incoming_point, incoming_vector, mirror_origin, plane_normal)

    reflected_ray = [incoming_ray[0], incoming_ray[1], list(intersect),
                     list(reflected_vector),
                     incoming_ray[4] + dist(incoming_point, intersect)]
    return reflected_ray


def raytrace_aperture(incoming_ray, aperture_origin, aperture_normal_vec,
                      aperture_range):
    # assume a circular aperture here.
    if incoming_ray is None:
        return None
    incoming_point = np.array(incoming_ray[2])
    incoming_vector = np.array(incoming_ray[3])
    # print(incoming_point, incoming_vector)

    aperture_normal = np.array(aperture_normal_vec)
    intersect, _ = reflect_line_about_plane(
        incoming_point, incoming_vector, aperture_origin, aperture_normal)

    if dist(intersect, aperture_origin) > aperture_range:
        return None
    # Otherwise keep the same vector as before and transmit through!
    through_ray = [incoming_ray[0], incoming_ray[1], list(intersect),
                   list(incoming_vector),
                   incoming_ray[4] + dist(incoming_point, intersect)]
    return through_ray


def TRTRioMPickle_mod(Ri, p1, p2, p3, p4, originM, mirror_coeffs,
                      return_all_rays=True):
    '''New routine for testing the new FTS style'''
    # The first rays start at around where mirror 10 is
    # Go through P1, then E8, then etc.
    Ray_TP1 = IntPolT2(Ri, coeffpolar, originpolar1, p1)  # P1
    Ray_E8 = ReflEll(Ray_TP1, thet6, origin8,
                     coeffellipse56, center8, range8)  # E8
    Ray_RP2 = IntPolR2(Ray_E8, coeffpolar, originpolar2, p2)  # P2
    Ray_E1 = ReflEll(Ray_RP2, thet, origin1,
                     coeffellipse, center1, range1)  # E3
    # Now reflect off the dihedral mirror!
    Ray_M0, Ray_MM = raytrace_dihedral_mirror(Ray_E1, mirror_coeffs, originM)

    Ray_E2 = ReflEll(Ray_MM, thet, origin2, coeffellipse,
                     center2, range2)  # off E4
    Ray_TP3 = IntPolT2(Ray_E2, coeffpolar, originpolar3, p3)  # P3
    Ray_E6 = ReflEll(Ray_TP3, thet6, origin6, coeffellipse56, center6, range6)
    Ray_RP4 = IntPolR2(Ray_E6, coeffpolar, originpolar4, p4)

    # Reflect off the left mirror now!
    Ray_E72 = ReflEll(Ray_RP4, thet7, origin7, coeffellipse7, center7, range7)
    if return_all_rays:
        return Ri, Ray_TP1, Ray_E8, Ray_RP2, Ray_E1, Ray_M0, Ray_MM, Ray_E2, Ray_TP3, Ray_E6, Ray_RP4, Ray_E72
    else:
        return Ray_E72


def run_ray_through_sim(initial_ray, config, mirror_position, path_order,
                        return_all_rays=False):
    '''New streamlined routine'''
    # The order is always polarizer, mirror, polarizer, mirror, dihedral,
    # mirror, polarizer, mirror, polarizer, mirror
    # the config file makes this very easy now!!!!!!!

    # path order needs to input a list of elements
    # on top of that the correct reflect/transmit options for the polarizers
    # So this one could be T, 8, R, 1, (mirror), 2, T, 6, R, 7
    # or [T, 8, R, 1, 2, T, 6, R, 7]

    # If we want to go a step further we know that transmitting first always
    # leads to mirror 8, and so this is redundant information. Can create a
    # hashmap always leading to the next aspect, but then we always have to
    # keep track of which side we're on... kind of a bit of a pain. maybe
    # later.
    function_map = {'T': IntPolT2, 'R': IntPolR2,
                    'DM': raytrace_dihedral_mirror, 'SM': IntM2,
                    'OM': raytrace_other_mirror, 'A': raytrace_aperture,
                    'E': ReflEll}

    ray = initial_ray
    all_rays = [ray]
    for instruction in path_order:
        function_key = instruction[:-1]
        item_number = int(instruction[-1])
        function = function_map[function_key]
        # See which of the polarizer, mirror, or central mirror we're going
        # through and update function arguments accordingly.
        if function_key in ['T', 'R']:  # Reflect or transmit off a polarizer
            # The equivalent of (coeffpolar, originpolar4, p4)
            args = (
                get_aspect(config, 'coefficients', 'polarizers', None),
                get_aspect(config, 'origins', 'polarizers', item_number),
                get_aspect(config, 'polarizer_values', None, item_number))

        # Reflect off the central dihedral mirror
        elif function_key in ['DM', 'SM']:
            args = (config['coefficients']['mirror'], mirror_position)

        elif function_key == 'OM':
            # reflect off one of the other focusing mirrors
            args = (config['other_mirrors'][item_number]['origin'],
                    config['other_mirrors'][item_number]['normal_vec'],
                    None)

        elif function_key == 'A':
            # reflect off one of the final focusing mirrors
            args = (config['apertures'][item_number]['origin'],
                    config['apertures'][item_number]['normal_vec'],
                    config['apertures'][item_number]['range'])
        else:  # Reflect off one of the ellipsoidal mirrors
            assert function_key == 'E'
            # The equivalent of (thet7, origin7, coeffellipse7, center7, range7)
            args = (
                get_aspect(config, 'angles', 'ellipses', item_number),
                get_aspect(config, 'origins', 'ellipses', item_number),
                get_aspect(config, 'coefficients', 'ellipses', item_number),
                get_aspect(config, 'centers', None, item_number),
                get_aspect(config, 'ranges', None, item_number))

        # raytrace_dihedral_mirror returns two rays instead of one!!
        if function_key == 'DM':
            ray_middle, ray = function(ray, *args)
            all_rays.append(ray_middle)
        else:
            ray = function(ray, *args)

        all_rays.append(ray)

    if return_all_rays:
        return all_rays
    else:
        return all_rays[-1]


def get_mirrors(instruction_set, polarizer_order=[1, 2, 3, 4]):
    # Have numbers for side1, side2
    polarizer_index = 0
    side1 = [9, 3, 4, 6, 7]
    side2 = [8, 1, 2, 5, None]
    sides = [side1, side2]

    pointer = 0
    # We start on side1
    side = 0
    values = []

    for instruction in instruction_set:
        if instruction == 'T':
            side = 1 - side
        value = sides[side][pointer]
        if instruction in ['T', 'R']:
            values.append(instruction + str(polarizer_order[polarizer_index]))
            polarizer_index += 1
        else:
            values.append(instruction + '0')
        # now add the mirror afterwards
        values.append('E' + str(value))
        pointer += 1
    return values


def get_possible_paths():
    instructions = [
        ['T', 'T', 'DM', 'T', 'R'],
        ['R', 'R', 'DM', 'R', 'T'],
        ['T', 'T', 'DM', 'R', 'T'],
        ['R', 'T', 'DM', 'T', 'T'],
        ['R', 'T', 'DM', 'R', 'R'],
        ['T', 'R', 'DM', 'R', 'R'],
        ['R', 'R', 'DM', 'T', 'R'],
        ['T', 'R', 'DM', 'T', 'T']
    ]

    # Don't include the last mirror here.
    possible_paths = [get_mirrors(instruc)[:-1] for instruc in instructions]
    return possible_paths


def run_rays_through_sim(initial_rays, config, mirror_position, paths=None):
    # Runs each ray through 8 possible paths
    output_rays = []
    if paths is None:
        paths = get_possible_paths()  # A little hard-coded here...
    for ray in initial_rays:
        for path in paths:
            output_ray = run_ray_through_sim(ray, config, mirror_position,
                                             path)
            # It it makes it though the sim, add to to the output list.
            if output_ray is not None:
                output_rays.append(output_ray)

    # Only return the rays that are within range of the final detector.
    final_rays = get_final_rays_tilt(
        output_rays, config['detector']['center'],
        config['detector']['range'], config['detector']['normal_vec'])

    return final_rays


def run_all_rays_through_sim(initial_rays, config, num_mirror_positions,
                             ymax=18, paths=None):

    delay = []
    all_final_rays = []  # added to capture final points on detector

    # Then for each y perform the rest of the raytracing
    for y in tqdm(np.linspace(-1 * ymax, ymax, int(num_mirror_positions))):
        # Run through all the possible paths
        mirror_position = list(
            np.array(config['origins']['mirror']) + [0, y, 0])

        final_rays = run_rays_through_sim(
            initial_rays, config, mirror_position, paths=paths)

        # should get rid of this magic number in delay calculation..
        delay.append(y * 4)
        all_final_rays.append(final_rays)

    return delay, all_final_rays


def separate_paths(paths, middle='M'):
    # separate before and after the 'M' or 'M2' feature
    paths1, paths2 = ([], [])

    for path in paths:
        middle_index = path.index(middle)
        paths1.append(path[:middle_index])
        paths2.append(path[middle_index:])
    return paths1, paths2


def run_rays_through_sim_first_half(initial_rays, config, paths):
    intermediate_rays = []
    for m in ['DM0', 'SM0']:
        if m in paths[0]:
            middle = m

    paths_first_half, paths_second_half = separate_paths(paths, middle=middle)

    for ray in initial_rays:
        for path1, path2 in zip(paths_first_half, paths_second_half):
            output_ray = run_ray_through_sim(ray, config, None, path1)
            if output_ray is not None:
                intermediate_rays.append({"ray": output_ray, "path": path2})

    return intermediate_rays


def run_rays_through_sim_second_half(intermediate_rays, config,
                                     mirror_position):
    output_rays = []
    for ray_and_path in intermediate_rays:
        ray = ray_and_path["ray"]
        path = ray_and_path["path"]
        output_ray = run_ray_through_sim(ray, config, mirror_position, path)
        if output_ray is not None:
            output_rays.append(output_ray)

    final_rays = get_final_rays_tilt(
        output_rays, config['detector']['center'], config['detector']['range'],
        config['detector']['normal_vec'])
    return final_rays


def run_all_rays_through_sim_optimized(
        initial_rays, config, num_mirror_positions, ymax=18, paths=None,
        progressbar=True):
    if paths is None:
        paths = get_possible_paths()  # A little hard-coded here...

    delay = []
    all_final_rays = []  # added to capture final points on detector

    intermediate_rays = run_rays_through_sim_first_half(initial_rays, config,
                                                        paths)

    # Then for each y perform the rest of the raytracing
    for y in tqdm(np.linspace(-1 * ymax, ymax, int(num_mirror_positions)),
                  disable=(not progressbar)):
        # Run through all the possible paths
        mirror_position = list(
            np.array(config['origins']['mirror']) + [0, y, 0])

        final_rays = run_rays_through_sim_second_half(intermediate_rays,
                                                      config, mirror_position)

        # should get rid of this magic number in delay calculation..
        delay.append(y * 4)
        all_final_rays.append(final_rays)

    return delay, all_final_rays


def test_eq(v1, v2, lim=.00000001):
    return (np.abs(np.subtract(v1, v2)) < lim).all()


def test_dims():
    assert test_eq(p1, np.pi/4, lim=.0001)
    assert test_eq(p2, np.pi/2, lim=.0001)
    assert test_eq(p3, np.pi, lim=.0001)
    assert test_eq(p4, np.pi/4, lim=.0001)

    assert test_eq(originG, [0., 0., 0.])  # the global origin
    # rotation with respect to itself aka 0,0,0
    assert test_eq(thetG, [0., 0., 0.])
    assert test_eq(origin1, [-32.075, -128., 0.])  # x,y (ellipse1)
    # x,y (ellipse2) #ELLIPSE DIMENSIONS:
    assert test_eq(origin2, [32.075, -128., 0.])
    assert test_eq(origin3, [-32.075, 128., 0.])  # x,y (ellipse3)
    assert test_eq(origin4, [32.075, 128, 0.])  # x,y  (ellipse4)

    if 'old' in filename:
        assert test_eq(origin5, [96.225, -120.50, 0.])  # (ellipse5)
        assert test_eq(origin6, [96.225, 120.50, 0.])  # (ellipse6)
        assert test_eq(origin7, [128.3, 7.5, 39.850000])  # (ellipse7)
        assert test_eq(origin8, [-96.225, -120.50, 0.])  # (ellipse8)
        assert test_eq(origin9, [-96.225, 120.50, 0.])  # (ellipse9)
        assert test_eq(origin10, [-128.3, 7.5, -39.850000])  # last ellipsoid
    else:
        assert test_eq(origin5, [96.225, -128., 0.])  # (ellipse5)
        assert test_eq(origin6, [96.225, 128., 0.])  # (ellipse6)
        assert test_eq(origin7, [160.375, 128., 0.])  # (ellipse7)
        assert test_eq(origin8, [-96.225, -128, 0.])  # (ellipse8)
        assert test_eq(origin9, [-96.225, 128, 0.])  # (ellipse9)
        assert test_eq(origin10, [-160.375, 128., 0.])  # last ellipsoid

    # for the four center ellipses
    assert test_eq(coeffellipse, [263.915180503, 256.0, 64.15])

    if 'old' in filename:
        # coefficients are like [a,b,c] of an ellipsoid x^2/a^2 + y^2/b^2 + z^2/c^2
        assert test_eq(coeffellipse7, [164.54585247700001,
                                       99.690818975602866, 130.9086635])  # for ellipse 7
        assert test_eq(coeffellipse56, [256.65344272795591, 248.39387505453516,
                                        64.58693753])  # for ellipses 5&6&8&9
    else:
        assert test_eq(coeffellipse7, coeffellipse)
        assert test_eq(coeffellipse56, coeffellipse)

    assert test_eq(coeffmirr, [31.75, 25.4, 19.05])  # for the mirror
    # for polarizers (2d circle)
    assert test_eq(coeffpolar, [32.075, 32.075, 0.])

    assert test_eq(thet, [0., 0., 0.])
    if 'old' in filename:
        assert test_eq(thet5, [0., 0., -.116385])  # angle of rotation
        assert test_eq(thet6, [0., 0., .116385])  # angle of rotation
        # angle of rotation
        assert test_eq(thet7, [0., 0.309319724356, 1.31064594453])
        # angle of rotation
        assert test_eq(thet10, [0., 0.309319724356, -1.31064594453])
    else:
        assert test_eq([thet5, thet6, thet7, thet10], [thet, thet, thet, thet])

    # global origin of polarizer 1
    assert test_eq(originpolar1, [-128.3, 0.0, 0.0])
    # global origin of polarizer 2
    assert test_eq(originpolar2, [-64.15, 0.0, 00])
    # global origin of polarizer 3
    assert test_eq(originpolar3, [64.15, 0.0, 0.0])
    # global origin of polarizer 4
    assert test_eq(originpolar4, [128.3, 0.0, 0.0])

    # below are the geometric centers and ranges of aspects of the FTS.
    assert test_eq([center1, range1], [[0.0, 254.99883040161689, 0.0], [
        31.427020202020202, 200, 31.400097443962792]])
    assert test_eq([center2, range2], [[-3.5527136788005009e-15, 254.99883040161689,
                                        0.0], [31.427020202020202, 200, 31.400097443962792]])
    assert test_eq([center3, range3], [[0.0, -254.99883040161689,
                                        0.0], [31.427020202020202, 200, 31.400097443962792]])
    assert test_eq([center4, range4], [[-3.5527136788005009e-15, -254.99883040161689,
                                        0.0], [31.427020202020202, 200, 31.400097443962792]])

    if 'old' in filename:
        assert test_eq([center5, range5], [[28.333333333333336, 244.95832095066169, 0.0], [
            30.909090909090921, 200, 31.315767057290568]])
        assert test_eq([center6, range6], [[28.333333333333336, -244.95832095066169,
                                            0.0], [30.909090909090921, 200, 31.315767057290568]])
        assert test_eq([center8, range8], [[-28.333333333333329, 244.95832095066169,
                                            0.0], [30.909090909090914, 200, 31.315767057290568]])
        assert test_eq([center9, range9], [[-28.333333333333329, -244.95832095066169,
                                            0.0], [30.909090909090914, 200, 31.315767057290568]])
        assert test_eq([center10, range10], [[-96.458686868686868, 3.279771448324329, 73.24759051407338], [
            41.577020202020229, 33.648217777841843, 20.001702052348694]])
        assert test_eq([center7, range7], [[96.45868686868684, 3.1907688264097978, -73.870253223692842], [
            41.577020202020208, 200, 19.642335814884394]])
    else:
        assert test_eq([center5, range5], [center1, range1])
        assert test_eq([center6, range6], [center3, range3])
        assert test_eq([center8, range8], [center1, range1])
        assert test_eq([center9, range9], [center3, range3])
        assert test_eq([center10, range10], [center3, range3])
        assert test_eq([center7, range7], [center3, range3])

    return True
