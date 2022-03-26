
Geometry.OCCTargetUnit = "CM";
Merge "stp/M5-CB-0.50mm.STEP";

epithelium_thickness = 0.005;

Physical Surface("cover") = {1};
Physical Surface("body") = {2};
Physical Curve("fixed") = {8, 9, 6};
Physical Curve("pressure") = {7};

// Mesh.MeshSizeMin = 0.001;
Field[1] = Distance;
Field[1].EdgesList = {7};
Field[1].NNodesByEdge = 150;

Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = 0.02;
Field[2].LcMax = 0.04;
Field[2].DistMin = epithelium_thickness;
Field[2].DistMax = 10*epithelium_thickness;

Background Field = 2;
Mesh.Smoothing = 10;

Mesh.CharacteristicLengthExtendFromBoundary = 0;
Mesh.CharacteristicLengthFromPoints = 0;
Mesh.CharacteristicLengthFromCurvature = 0;
