#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:52:36 2020

@author: Thomas Camminady
"""

import numpy as np
import warnings


def getVerticesAndFaces(platonicsolid):

    if platonicsolid in ["4", 4, "tetra", "Tetra", "tetrahedron", "Tetrahedron"]:
        return tetrahedron()
    elif platonicsolid in ["8", 8, "octa", "Octa", "octahedron", "Octahedron"]:
        return octahedron()
    elif platonicsolid in ["20", 20, "ico", "Ico", "icosahedron", "Icosahedron"]:
        return icosahedron()
    else:
        warnings.warn(
            "platonicsolid has to be Octahedron or Icosahedron,"
            + " but you chose {}".format(platonicsolid)
            + ". We will return the Octahedron quadrature."
        )


def tetrahedron():
    vertices = np.array(
        [
            [np.sqrt(8 / 9), 0, -1 / 3],
            [-np.sqrt(2 / 9), np.sqrt(2 / 3), -1 / 3],
            [-np.sqrt(2 / 9), -np.sqrt(2 / 3), -1 / 3],
            [0, 0, 1.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    return vertices, faces


def octahedron():
    vertices = np.array(
        [
            [1.0, 0, 0],
            [0, 1.0, 0],
            [0, 0, 1.0],
            [-1.0, 0, 0],
            [0, -1.0, 0],
            [0, 0, -1.0],
        ]
    )

    faces = np.array(
        [
            [0, 1, 2],
            [1, 2, 3],
            [0, 2, 4],
            [2, 3, 4],
            [0, 1, 5],
            [0, 4, 5],
            [1, 3, 5],
            [3, 4, 5],
        ]
    )
    return vertices, faces


def icosahedron():
    r = (1 + np.sqrt(5)) / 2.0  # golden ratio
    vertices = np.array(
        [
            [0, 1, r],
            [0, 1, -r],
            [0, -1, r],
            [0, -1, -r],
            [1, r, 0],
            [1, -r, 0],
            [-1, r, 0],
            [-1, -r, 0],
            [r, 0, 1],
            [r, 0, -1],
            [-r, 0, 1],
            [-r, 0, -1],
        ]
    )
    for i in range(vertices.shape[0]):
        vertices[i, :] /= np.linalg.norm(vertices[i, :])

    faces = (
        np.array(
            [
                [1, 3, 9],
                [1, 3, 11],
                [1, 5, 7],
                [1, 7, 11],
                [2, 5, 7],
                [2, 12, 7],
                [2, 5, 10],
                [2, 4, 10],
                [2, 4, 12],
                [11, 8, 12],
                [4, 8, 12],
                [4, 8, 6],
                [4, 6, 10],
                [6, 10, 9],
                [5, 10, 9],
                [1, 9, 5],
                [7, 11, 12],
                [3, 9, 6],
                [3, 8, 6],
                [3, 8, 11],
            ]
        )
        - 1
    )

    return vertices, faces
