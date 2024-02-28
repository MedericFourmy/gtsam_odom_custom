- custom_odom_factors.py: implements 2 error functions, integration of a local twist or global twist
- example_custom_odom.py: toy example of constant velocity integration with absolute pose measurements 
Parameters to tweak:
    - LOCAL_TWIST: whether to use the local or global twist formulation
    - noisy_meas: applies some noise to ground truth to simulate measurements
    - noisy_init: initialize away from the ground-truth
- test_numdiff.py: check analytical jacobians against numdiff (copy pasted formulas, not great but ok)
- test_gtsam_vs_pin.py: sanity checks between pinocchio and gtsam
