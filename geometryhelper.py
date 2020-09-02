#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 12:23:29 2020

@author: Thomas Camminady
"""
import numpy as np
from numba import njit


@njit
def distance(pointa, pointb):
    """
    Returns the spherical distance between
    two points.
    """

    ra, rb = np.linalg.norm(pointa), np.linalg.norm(pointb)
    return ra * np.arccos(np.dot(pointa, pointb) / ra ** 2)


@njit
def angle(pointb, pointa, pointc):
    """
    Returns the spherical angle between the lines
    pointb<->pointa and pointa<->pointc
    https://en.wikipedia.org/wiki/Spherical_trigonometry#Cosine_rules_and_sine_rules .
    """

    c = distance(pointb, pointa)
    b = distance(pointa, pointc)
    a = distance(pointc, pointb)
    cosangle = (np.cos(a) - np.cos(b) * np.cos(c)) / (np.sin(b) * np.sin(c))
    return np.arccos(cosangle)


@njit
def getarea(pointa, pointb, pointc):
    """ "
    Returns the spherical area of the triangle
    spanned by the three points, see:
    https://en.wikipedia.org/wiki/Spherical_trigonometry#Area_and_spherical_excess .
    """

    ra, rb, rc = np.linalg.norm(pointa), np.linalg.norm(pointb), np.linalg.norm(pointc)

    pointa /= ra
    pointb /= rb
    pointc /= rc

    alpha = angle(pointb, pointa, pointc)
    beta = angle(pointc, pointb, pointa)
    gamma = angle(pointa, pointc, pointb)
    return (alpha + beta + gamma - np.pi) * ra ** 2
