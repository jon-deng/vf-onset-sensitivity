// Gmsh project created on Thu Mar 31 08:55:14 2022

Point(1) = {0, 0, 0};
Point(2) = {0.5, 0, 0};
Point(3) = {0.5, 0.5, 0};
Point(4) = {0, 0.5, 0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {3, 4, 1, 2};

Surface(1) = {1};

Physical Surface("body") = {1};
Physical Curve("fixed") = {1};
Physical Curve("pressure") = {2, 3, 4};

Transfinite Surface {1} = {1, 2, 3, 4};
