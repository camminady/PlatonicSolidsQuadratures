# PlatonicSolidsQuadratures
Geometric quadrature sets, based upon a triangulation of some Platonic solids.

## How To?
```python
from platonicsolidsquadrature import platonicsolidsquadrature
points,weights,neighbours,dual,platonicpoints,platonicneighbours =  platonicsolidsquadrature("ico",10)
```

For example, we can choose the icosahedron as the platonic solid.
Quadrature points and weights result from triangulating the faces of an icosahedron. Each quadrature point has six neighbours (except for the poles, they have five). Connecting the centers of the six surrounding triangles yields a hexagon. That hexagon is the quadrature weight of a quadrature point, shown below where the color represents the quadrature weight.

![Base geometry and icoshaedron quadrature with weights.](https://raw.githubusercontent.com/camminady/PlatonicSolidsQuadratures/master/image1.png)
