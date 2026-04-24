"""
Synergy Predictor V3 — Pre-merge prediction of merging effect.

Uses parameter-space and functional features that are available
BEFORE running the merge, unlike V1/V2 which required the NxN matrix.

Three feature families:
A. Parameter conflict (sign agreement, magnitude ratios)
B. Subspace overlap (Task Singular Vectors via SVD)
C. Prediction agreement (source consensus on unlabeled target data)
"""
