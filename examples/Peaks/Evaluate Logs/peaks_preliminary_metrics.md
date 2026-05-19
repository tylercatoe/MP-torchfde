# Peaks Initial Tests Metrics Summary

``` text
------------------------------  Peaks Tests  ------------------------------

------------------------------
Direct:
------------------------------
Epoch 000 | Time 0.80s | Peak Mem 1060.98 MB | LR 9.0460e-02 | Train MSE 12.6966 | Test MSE 12.7169 | Best Test MSE 12.7169
Epoch 001 | Time 0.44s | Peak Mem 1060.98 MB | LR 6.5485e-02 | Train MSE 5.9929 | Test MSE 5.9988 | Best Test MSE 5.9988
Epoch 002 | Time 0.44s | Peak Mem 1060.98 MB | LR 3.4615e-02 | Train MSE 3.9125 | Test MSE 3.8797 | Best Test MSE 3.8797
Epoch 003 | Time 0.45s | Peak Mem 1060.98 MB | LR 9.6396e-03 | Train MSE 2.9983 | Test MSE 3.0145 | Best Test MSE 3.0145
Epoch 004 | Time 0.45s | Peak Mem 1060.98 MB | LR 1.0000e-04 | Train MSE 2.8560 | Test MSE 2.8770 | Best Test MSE 2.8770

------------------------------
Standard adjoint:
------------------------------
Epoch 000 | Time 0.83s | Peak Mem 712.22 MB | LR 9.0460e-02 | Train MSE 3.6858 | Test MSE 3.6962 | Best Test MSE 3.6962
Epoch 001 | Time 0.46s | Peak Mem 712.22 MB | LR 6.5485e-02 | Train MSE 18.8738 | Test MSE 18.8567 | Best Test MSE 3.6962
Epoch 002 | Time 0.47s | Peak Mem 712.22 MB | LR 3.4615e-02 | Train MSE 5.0433 | Test MSE 5.0493 | Best Test MSE 3.6962
Epoch 003 | Time 0.46s | Peak Mem 712.22 MB | LR 9.6396e-03 | Train MSE 4.8976 | Test MSE 4.8976 | Best Test MSE 3.6962
Epoch 004 | Time 0.46s | Peak Mem 712.22 MB | LR 1.0000e-04 | Train MSE 3.7500 | Test MSE 3.7539 | Best Test MSE 3.6962

------------------------------
Adjoint MP with float16:
------------------------------
Epoch 000 | Time 1.17s | Peak Mem 484.33 MB | LR 9.0460e-02 | Train MSE 21.3645 | Test MSE 21.2276 | Best Test MSE 21.2276
Epoch 001 | Time 0.48s | Peak Mem 484.33 MB | LR 6.5485e-02 | Train MSE 13.6650 | Test MSE 13.7133 | Best Test MSE 13.7133
Epoch 002 | Time 0.51s | Peak Mem 484.33 MB | LR 3.4615e-02 | Train MSE 5.8199 | Test MSE 5.8578 | Best Test MSE 5.8578
Epoch 003 | Time 0.76s | Peak Mem 484.33 MB | LR 9.6396e-03 | Train MSE 4.8987 | Test MSE 4.8921 | Best Test MSE 4.8921
Epoch 004 | Time 0.91s | Peak Mem 484.33 MB | LR 1.0000e-04 | Train MSE 3.3197 | Test MSE 3.3252 | Best Test MSE 3.3252

------------------------------
Adjoint MP with bfloat16:
------------------------------
Epoch 000 | Time 0.81s | Peak Mem 480.20 MB | LR 9.0460e-02 | Train MSE 9.3440 | Test MSE 9.3180 | Best Test MSE 9.3180
Epoch 001 | Time 0.38s | Peak Mem 480.20 MB | LR 6.5485e-02 | Train MSE 6.4136 | Test MSE 6.4218 | Best Test MSE 6.4218
Epoch 002 | Time 0.38s | Peak Mem 480.20 MB | LR 3.4615e-02 | Train MSE 8.7782 | Test MSE 8.7919 | Best Test MSE 6.4218
Epoch 003 | Time 0.41s | Peak Mem 480.20 MB | LR 9.6396e-03 | Train MSE 6.0009 | Test MSE 6.0014 | Best Test MSE 6.0014
Epoch 004 | Time 0.38s | Peak Mem 480.20 MB | LR 1.0000e-04 | Train MSE 2.8789 | Test MSE 2.8813 | Best Test MSE 2.8813
```


Experiment Parameters:
- Network Architecture:
    - Width: 256
    - Input layer -> tanh() -> FDE_Block -> Output layer 
- FDE_Block:
    - Beta: 0.5
    - T: 2.0
    - step_size: 0.1
    - $f$ in $D^\beta z = f$: 3 layer MLP
- Training Arguments:
    - Epochs: 5
    - Batch size: 10,000
    - Total samples: 100,000
    - Initial LR: 0.1
    - Weight decay: 5e-4
    - GPU: NVIDIA A100 (Colab)

Note: 
- adjoint mode uses adjoint method for gradients but in high precision
- adjoint-mixed mode uses adjoint method with float16 for mixed precision (and hence the DynamicScaler)
- adjoint-mixed-bflat uses adjoint method with bfloat16 for mixed precision (and hence no DynamicScaler)
- direct mode uses standard backprop with high precision