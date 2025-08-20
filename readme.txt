Extrusion Film Casting
efc_pinn_adaptive_multi-seed.py

This code runs the PINNs method with the adaptive activation function tanh  to solve the EFC equations with a fixed set of parameters for multiple seeds. It plots a comparison between solutions (averaged over the seeds) obtained by the model and numerical methods.

efc.py
The complete pipeline for training and evaluation. Also plots a comparison between solutions (averaged over mutliple seeds) obtained by the model and numerical methods.
-----------------------------------------------------------------------------------------------------------------------------------------
Fibrespinning
fs_multi_PINN.py

PINN for solving the steady state fibre spinning equations. Architecture comprises 3 PINNs trained sequentially with randomized grid search + over multiple seeds and then averaged.
