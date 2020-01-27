import logging
from builtins import range, int

import numpy as np
import math
import copy
# import geopandas

from collections import namedtuple
from pprint import pprint as pp
import sys

# package modules
import aeolis.wind

# from aeolis.utils import *

# initialize logger
logger = logging.getLogger(__name__)


def initialize(s, p):
    fence_height = 1


    # generate new variables to define fences
    # fences = geopandas.read_file('fences.shp') #eventually would like to add in more functionality here
    s['fence_base'] = copy.copy(s['zb'])  # initial fence base is the bed elevation
    s['fence_top'] = copy.copy(s['fence_base'])  # initialize all fence tops as zero and fill in after

    # populate fence tops - right now this is picking same point in domain every time
    ileny = np.round(np.size(s['fence_top'], 0) / 2)
    if ileny > 1:
        ileny = np.array([ileny - 5, ileny - 4, ileny - 3, ileny - 2, ileny - 1, ileny])
    ix = int(np.round(np.size(s['fence_top'], 1) / 2))
    for j in range(np.size(ileny)):
        iy = int(ileny[j])
        s['fence_top'][iy][ix] = s['fence_top'][iy][ix] + fence_height

    # populate fence tops - right now this is picking same point in domain every time
    ileny = np.round(np.size(s['fence_top'], 0) / 3)
    if ileny > 1:
        ileny = np.array([ileny - 5, ileny - 4, ileny - 3, ileny - 2, ileny - 1, ileny])
    ix = int(np.round(np.size(s['fence_top'], 1) / 3))
    for j in range(np.size(ileny)):
        iy = int(ileny[j])
        s['fence_top'][iy][ix] = s['fence_top'][iy][ix] + fence_height


    # populate fence tops - right now this is picking same point in domain every time
    ileny = np.round(np.size(s['fence_top'], 0) / 4)
    if ileny > 1:
        ileny = np.array([ileny - 5, ileny - 4, ileny - 3, ileny - 2, ileny - 1, ileny])
    ix = int(np.round(np.size(s['fence_top'], 1) / 8))
    for j in range(np.size(ileny)):
        iy = int(ileny[j])
        s['fence_top'][iy][ix] = s['fence_top'][iy][ix] + fence_height

    s['fence_height'] = s['fence_top'] - s['fence_base']
    s['fence_height_init'] = s['fence_height']

    return s


def update_fence_height(s, p):
    # if hasattr(s, 'zbold'):
    dz = s['zb'] - s['zbold']
    # else:
    # dz = s['zb']*0

    fence_base = s['fence_base'] + dz
    fence_height = s['fence_top'] - fence_base

    # if initial fence height was zero it should be zero here
    ix = s['fence_height_init'] < 0.01
    fence_height[ix] = 0

    # ensure no sub-zero values
    ix = fence_height < 0.01
    fence_height[ix] = 0

    return fence_height


def fence_shear(s, p):
    nx = p['nx'] + 1
    ny = p['ny'] + 1

    # nx = p['nx']
    # ny = p['ny']

    tau = copy.copy(s['tau'])
    gradx = np.gradient(s['x'], axis=0)
    grady = np.gradient(s['y'], axis=1)
    shear_red = copy.copy(s['fence_height']) * 0  # allocate new variable to accumulate shear reductions

    fence_height = copy.copy(s['fence_height'])
    x = copy.copy(s['x'])
    y = copy.copy(s['y'])

    #udir = copy.copy(s['alfa'][1, 1]) + copy.copy(s['udir'][1, 1])
    udir = copy.copy(s['udir'][1, 1])

    fencelocs = np.where(s['fence_height'] > 0)

    # [n for n, i in enumerate(s['fence_height']) if i > 0.7][0]

    # fencelocs = [i for i, x in enumerate(s['fence_height']) if x >0]


    # loop through each x location and each fence location to calculate wind shear reduction from fences
    for jj in range(nx):
        for ii in range(ny):

            height = fence_height[ii][jj] #should also store fence ID somewhere to make sure that are not double counting

            if height > 0:  # only proceed if there is fence there

                dist_matrix = np.sqrt((x - x[ii][jj]) ** 2 + (y - y[ii][jj]) ** 2)

                # now loop through again to determine if new fence is in path
                for j in range(nx):
                    for i in range(ny):

                        if (0.1273 * dist_matrix[i, j]) / height < 0.7: #only proceed if distance criteria is satisfied

                            #define bounds of grid cell looking at
                            xlims = [x[i, j] - gradx[i, j] / 2, x[i, j] + gradx[i, j] / 2]
                            ylims = [y[i, j] - grady[i, j] / 2, y[i, j] - grady[i, j] / 2]
                            if np.abs(udir) <= 45:  # look at the left face of the cell
                                xs = [xlims[0], xlims[0]]
                                ys = [ylims[0], xlims[1]]
                            elif udir <= 135 and udir > 45:  # look at the bottom face of the cell
                                xs = [xlims[0], xlims[1]]
                                ys = [ylims[1], xlims[1]]
                            elif udir < -45 and udir >= -135:  # look at the top face of the cell
                                xs = [xlims[0], xlims[1]]
                                ys = [ylims[1], xlims[1]]
                            else:  # look at the right face of the cell
                                xs = [xlims[1], xlims[1]]
                                ys = [ylims[0], xlims[1]]


                            #define fence location and downwind cells
                            fencex = x[ii, jj]
                            fencey = y[ii, jj]
                            if np.abs(udir) < 90:
                                xf = [fencex, np.max(x)]
                                yf = [fencey, fencey + (xf[1] - xf[0]) * np.tan(udir / 180 * np.pi)]
                            elif np.abs(udir) == 90:
                                xf = [fencex, fencex]
                                yf = [fencey, np.max(y)]
                            elif np.abs(udir) == -90:
                                xf = [fencex, fencex]
                                yf = [fencey, np.min(y)]
                            else:
                                xf = [fencex, np.min(x)]
                                yf = [fencey, fencey + (xf[0] - xf[1]) * np.tan(udir / 180 * np.pi)]

                            #find if there is an intersection between region downwind of fence and the cell in question
                            xi, yi, val, r, s = intersectLines([xs[0], ys[0]], [xs[1], ys[1]], [xf[0], yf[0]],
                                                               [xf[1], yf[1]])

                            if np.size(xi) > 0: #only proceed if there is an intersection point

                                #seems to be an error somewhere so implementing simple fix, should only alter cells downwind of fence
                                if np.abs(udir) < 90 and x[i][j] >= x[ii, jj]:
                                    dist = dist_matrix[i, j]
                                    wind_reduce = 0.7 - (0.1273 * dist) / height
                                elif np.abs(udir) == 90 and y[i][j] > y[ii, jj]:
                                    dist = dist_matrix[i, j]
                                    wind_reduce = 0.7 - (0.1273 * dist) / height
                                elif np.abs(udir) == -90 and y[i][j] <= y[ii, jj]:
                                    dist = dist_matrix[i, j]
                                    wind_reduce = 0.7 - (0.1273 * dist) / height
                                elif np.abs(udir) > 90 and x[i][j] <= x[ii, jj]:
                                    dist = dist_matrix[i, j]
                                    wind_reduce = 0.7 - (0.1273 * dist) / height
                                else:
                                    wind_reduce = 0

                                #dist = np.sqrt((fencex - x[i][j]) ** 2 + (fencey - y[i][j]) ** 2)
                                dist = dist_matrix[i, j]

                                #print(dist)
                                #print(height)

                                if wind_reduce > 0.7:
                                    wind_reduce = 0.7
                                elif wind_reduce < 0:
                                    wind_reduce = 0

                                shear_red[i, j] = shear_red[i, j] + wind_reduce

    ix = shear_red > 1
    shear_red[ix] = 1

    # update tau
    taufence = tau - shear_red * tau
    np.savetxt('taufence.txt', taufence)
    np.savetxt('tau.txt', tau)
    np.savetxt('shearred.txt', shear_red)

    tau = taufence
    taus = taufence * np.cos(udir / 180. * np.pi)
    taun = taufence * np.sin(udir / 180. * np.pi)

    return tau, taus, taun


def intersectLines(pt1, pt2, ptA, ptB):
    """ Note, from: https://www.cs.hmc.edu/ACM/lectures/intersections.html

        this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)

        returns a tuple: (xi, yi, valid, r, s), where
        (xi, yi) is the intersection
        r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
        s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
            valid == 0 if there are 0 or inf. intersections (invalid)
            valid == 1 if it has a unique intersection ON the segment    """

    DET_TOLERANCE = 0.00000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1;
    x2, y2 = pt2
    dx1 = x2 - x1;
    dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA;
    xB, yB = ptB;
    dx = xB - x;
    dy = yB - y;

    # we need to find the (typically unique) values of r and s
    # that will satisfy
    #
    # (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
    #
    # which is the same as
    #
    #    [ dx1  -dx ][ r ] = [ x-x1 ]
    #    [ dy1  -dy ][ s ] = [ y-y1 ]
    #
    # whose solution is
    #
    #    [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
    #    [ s ] = DET  [ -dy1  dx1 ] [ y-y1 ]
    #
    # where DET = (-dx1 * dy + dy1 * dx)
    #
    # if DET is too small, they're parallel
    #
    DET = (-dx1 * dy + dy1 * dx)

    if math.fabs(DET) < DET_TOLERANCE: return (0, 0, 0, 0, 0)

    # now, the determinant should be OK
    DETinv = 1.0 / DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy * (x - x1) + dx * (y - y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))

    # return the average of the two descriptions
    xi = (x1 + r * dx1 + x + s * dx) / 2.0
    yi = (y1 + r * dy1 + y + s * dy) / 2.0
    return (xi, yi, 1, r, s)


def rayintersectseg(p, edge):
    ''' takes a point p=Pt() and an edge of two endpoints a,b=Pt() of a line segment returns boolean
    from: https://rosettacode.org/wiki/Ray-casting_algorithm
    '''
    a, b = edge
    if a.y > b.y:
        a, b = b, a
    if p.y == a.y or p.y == b.y:
        p = Pt(p.x, p.y + _eps)

    intersect = False

    if (p.y > b.y or p.y < a.y) or (
            p.x > max(a.x, b.x)):
        return False

    if p.x < min(a.x, b.x):
        intersect = True
    else:
        if abs(a.x - b.x) > _tiny:
            m_red = (b.y - a.y) / float(b.x - a.x)
        else:
            m_red = _huge
        if abs(a.x - p.x) > _tiny:
            m_blue = (p.y - a.y) / float(p.x - a.x)
        else:
            m_blue = _huge
        intersect = m_blue >= m_red
    return intersect
