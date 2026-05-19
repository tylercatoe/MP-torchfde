# STL10 Initial Tests Metrics Summary

```text
------------------------------  STL10 Tests  ------------------------------

------------------------------
Direct
------------------------------
Epoch 001 | Time 70.55s | Peak Mem 61898.69 MB | LR 7.5025e-02 | Train Acc 0.1042 | Val Acc 0.0830 | Best 0.0830
Epoch 002 | Time 70.25s | Peak Mem 61898.69 MB | LR 2.5075e-02 | Train Acc 0.1143 | Val Acc 0.1060 | Best 0.1060
Epoch 003 | Time 70.26s | Peak Mem 61898.69 MB | LR 1.0000e-04 | Train Acc 0.1653 | Val Acc 0.1500 | Best 0.1500

------------------------------
Adjoint
------------------------------
Epoch 001 | Time 79.12s | Peak Mem 36430.47 MB | LR 7.5025e-02 | Train Acc 0.1042 | Val Acc 0.0830 | Best 0.0830
Epoch 002 | Time 76.97s | Peak Mem 36430.47 MB | LR 2.5075e-02 | Train Acc 0.1022 | Val Acc 0.0910 | Best 0.0910
Epoch 003 | Time 77.06s | Peak Mem 36430.47 MB | LR 1.0000e-04 | Train Acc 0.1030 | Val Acc 0.0800 | Best 0.0910

------------------------------
Adjoint MP with float16
------------------------------
Epoch 001 | Time 124.14s | Peak Mem 24417.16 MB | LR 7.5025e-02 | Train Acc 0.1042 | Val Acc 0.0830 | Best 0.0830
Epoch 002 | Time 120.55s | Peak Mem 24417.16 MB | LR 2.5075e-02 | Train Acc 0.1022 | Val Acc 0.0910 | Best 0.0910
Epoch 003 | Time 121.66s | Peak Mem 24417.16 MB | LR 1.0000e-04 | Train Acc 0.1042 | Val Acc 0.0800 | Best 0.0910

------------------------------
Adjoint MP with bfloat16
------------------------------
Epoch 001 | Time 65.06s | Peak Mem 24155.22 MB | LR 7.5025e-02 | Train Acc 0.1042 | Val Acc 0.0830 | Best 0.0830
Epoch 002 | Time 63.37s | Peak Mem 24155.22 MB | LR 2.5075e-02 | Train Acc 0.1022 | Val Acc 0.0910 | Best 0.0910
Epoch 003 | Time 63.32s | Peak Mem 24155.22 MB | LR 1.0000e-04 | Train Acc 0.1042 | Val Acc 0.0880 | Best 0.0910
```

Experiment Parameters:
- Network Architecture:
    - Same as Lars but with FDE blocks instead of ODE blocks

- FDE_Block:
    - Beta: 0.5
    - T: 2.0
    - step_size: 0.1
    - $f$ in $D^\beta z = f$: Time-dependent dynamics with piecewise-constant weights (same as Lars')

- Training Arguments:
    - Downsampling and other things exactly same as Lars
    - Epochs: 3 just for preliminary smoke tests
    - Batch Size: 128
    - Initial LR: 0.1
    - Momentum: 0.9
    - GPU: NVIDIA A100 (Colab)

Note: 
- adjoint mode uses adjoint method for gradients but in high precision
- adjoint-mixed mode uses adjoint method with float16 for mixed precision (and hence the DynamicScaler)
- adjoint-mixed-bflat uses adjoint method with bfloat16 for mixed precision (and hence no DynamicScaler)
- direct mode uses standard backprop with high precision