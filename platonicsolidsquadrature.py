#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 12:20:20 2020

@author: Thomas Camminady
"""

import scipy
import numpy as np
from numba import njit

from platonicsolids import getVerticesAndFaces
from geometryhelper import getarea


def platonicsolidsquadrature(platonicsolid, n, slerpflag=False):

    if slerpflag:

        def interp(pointa, pointb, n):
            """Spherical linear interpolation between
            two points."""
            if n == 1:
                return pointa
            omega = np.arccos(np.dot(pointa, pointb))
            t = np.linspace(0, 1, n)

            return (
                np.outer(pointa, np.sin((1 - t) * omega) / np.sin(omega))
                + np.outer(pointb, np.sin((t) * omega) / np.sin(omega))
            ).T

    else:

        def interp(p0, p1, n):
            print(n)
            print(p0.shape)
            return np.linspace(p0, p1, n)

    vertices, faces = getVerticesAndFaces(platonicsolid)
    pts, con = getpoints(vertices, faces, n, interp)
    neighbours = getneighbours(pts, con)
    neighbours = sortneighbours(neighbours)
    weights, ptsdual = computeareaanddual(pts, neighbours)

    return pts, weights, neighbours, ptsdual, vertices, faces


def triangulateface(p0, p1, p2, n, interp):
    """
    Ginven three points and the number of points along an edge,
    this performs the triangulation and returns
    the new points and their connectivity. The connectivity
    is stored in an undirected way.
    """
    assert n >= 2

    def count(i):
        return (i + 1) * (i + 2) // 2  # Number of points up to row i.

    pts = np.zeros((0, 3))  # Store points' coordinates.
    con = np.zeros((0, 2), dtype=int)  # Store connection ids.

    p0p1 = interp(p0, p1, n)  # Interpolate along left edge of triangle.
    p0p2 = interp(p0, p2, n)  # Interpolate along right edge of triangle.

    for i in range(n):
        # Interpolate between left and right points.
        pts = np.vstack([pts, interp(p0p1[i, :], p0p2[i, :], i + 1)])

        # Points in the same horizontal row are connected.
        idsthisrow = np.arange(count(i - 1), count(i))
        con = np.vstack([con, np.array([idsthisrow[:-1], idsthisrow[1:]]).T])

        # Except for the last row, connect with the points from the next row.
        if not i == (n - 1):
            idsnextrow = np.arange(count(i), count(i + 1))
            # Down and left direction.
            con = np.vstack([con, np.array([idsthisrow, idsnextrow[:-1]]).T])
            # Down and right direction.
            con = np.vstack([con, np.array([idsthisrow, idsnextrow[1:]]).T])

    return pts, con


def triangulateallfaces(vertices, faces, n, interp):

    pts = np.zeros((0, 3))  # Store points' coordinates.
    con = np.zeros((0, 2), dtype=int)  # Store connection ids.
    for i in range(faces.shape[0]):
        a, b, c = faces[i, :]
        ptsi, coni = triangulateface(
            vertices[a, :], vertices[b, :], vertices[c, :], n, interp
        )
        con = np.vstack([con, coni + pts.shape[0]])  # Offset points ids and store.
        pts = np.vstack([pts, ptsi])  # Store new points

    return pts, con


@njit
def getmatches(pts):
    npts = pts.shape[0]
    matches = np.array(
        [
            (i, j)
            for i in range(npts)
            for j in range(i)
            if np.linalg.norm(pts[i, :] - pts[j, :]) < 1e-10
        ]
    )
    return matches


def merge(pts, con):
    """
    Some points are duplicates, we've got to merge them.
    Big overkill, but we'll compute this by doing a pairwise distance
    between any two points. If the distance is almost zero, the duplicate
    is removed and the connection entry is updated
    """

    matches = getmatches(pts)

    matches = np.sort(matches)
    newids = np.arange(pts.shape[0])
    toremove = np.zeros(0, dtype=int)
    for i in range(matches.shape[0]):
        if matches[i, 0] != matches[i, 1]:
            newids[matches[i, 1]] = min(
                newids[matches[i, 0]], matches[i, 1]
            )  # Remap to lower id.
            if not matches[i, 1] in toremove:
                toremove = np.append(toremove, matches[i, 1])

    # Remove gaps
    reduce = np.arange(np.max(newids) + 1)
    reduce[np.sort(np.unique(newids))] = np.arange(np.unique(newids).shape[0])
    newids = reduce[newids]

    con = newids[con]
    pts = np.delete(pts, toremove, axis=0)
    return pts, con


def getpoints(vertices, faces, n, interp):
    """
    Returns the unprojected points and their connectivity.
    npoints = 4n**2−8n+6
    """
    pts, con = triangulateallfaces(vertices, faces, n, interp)
    pts, con = merge(pts, con)
    return pts, con


def getneighbours(pts, con):
    ncons, npts = con.shape[0], pts.shape[0]
    C = scipy.sparse.coo_matrix(
        (np.ones(ncons), (con[:, 0], con[:, 1])), shape=(npts, npts)
    ).todense()
    C += C.T  # Make symmetric matrix.
    neighbours = np.nan * np.ones((npts, 6), dtype=int)
    for i in range(npts):
        neighbi = C[i, :].nonzero()[1]
        neighbours[i, : neighbi.shape[0]] = neighbi
    neighbours[np.isnan(neighbours)] = -999
    return neighbours.astype(int)


def sortneighbours(neighbours):
    """
    It will be advantageous to have the neighbours
    of a node ordered in a clockwise (or ccw) manner.
    This method achieves that.
    """
    n = neighbours.shape[0]
    nneighbours = [int(np.sum(neighbours[i, :] >= 0)) for i in range(n)]

    for i in range(n):
        unsorted = neighbours[i, : nneighbours[i]]  # Get unsorted neighbours.
        sorted = [
            unsorted[0]
        ]  # This will later be the sorted list of neighbours, start with the first.

        while len(sorted) < nneighbours[i]:
            # Get the neighbours of the last element in sorted.
            tmp = neighbours[sorted[-1], : nneighbours[sorted[-1]]]
            # Get the shared and not yet stored neighbours.
            shared = np.setdiff1d(np.intersect1d(unsorted, tmp), sorted)
            sorted.append(shared[0])

        neighbours[i, : nneighbours[i]] = sorted  # Store result.
    return neighbours


def computeareaanddual(pts, neighbours):
    nneighbours = [int(sum(neighbours[i, :] >= 0)) for i in range(neighbours.shape[0])]
    for i in range(pts.shape[0]):
        pts[i, :] /= np.linalg.norm(pts[i, :])

    ptsdual = np.nan * np.ones((*neighbours.shape, 3))
    areas = np.zeros((neighbours.shape[0]))
    for i in range(neighbours.shape[0]):
        x = neighbours[
            i, : nneighbours[i]
        ]  # the ids of the points which surround the i-th point
        for m, (j, k) in enumerate(
            zip(x, np.roll(x, 1))
        ):  # i,j,k form the m-th triangle
            pi, pj, pk = pts[i, :], pts[j, :], pts[k, :]
            pij = (pi + pj) / 2
            pik = (pi + pk) / 2
            pc = (pi + pj + pk) / 3
            areas[i] += getarea(pi, pij, pc) + getarea(pi, pc, pik)
            pcenter = (pi + pj + pk) / 3
            ptsdual[i, m, :] = pcenter
    return areas, ptsdual
