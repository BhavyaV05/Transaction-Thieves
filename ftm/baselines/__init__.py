"""baselines: GFP+LightGBM per bank and full graph; Multi-GNN, FraudGT, MEGA-GNN; DP-FedAvg GNN and one AML-specific federated method.

Owner: T1 (Abhishek).  Component: C1, C1b, Group A/B.  Tier: Core (Multi-GNN, DP-FedAvg) / Secondary.
Spec: plan.md 6.1-6.2, 6.17; tracks.md 3.4, 3.5, 3.7.

First steps:
  - gfp_gbt.py: reproduce F1 63.23 on HI-Small, then per-bank runs
  - multi_gnn.py: reproduce paper on its split, then our split
  - isotonic calibration on validation for every model
"""
