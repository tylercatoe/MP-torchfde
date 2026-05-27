# Peaks Final Metrics Summary

```text
mode                 | final_mse | best_mse | train_mem_mb | train_time_s | infer_time_s | infer_mem_mb
---------------------+-----------+----------+--------------+--------------+--------------+-------------
adjoint              | 0.00011   | 0.00011  | 708.58       | 6262.00      | 0.0100       | 56.20       
adjoint-mixed        | 0.00028   | 0.00028  | 483.23       | 4823.64      | 0.0646       | 45.82       
adjoint-mixed-bfloat | 0.000914  | 0.000687 | 479.10       | 3432.30      | 0.0435       | 45.82       
direct               | 0.0001    | 0.0001   | 1038.70      | 5804.10      | 0.0232       | 64.99       
```

Log files:
- adjoint: adj_full_training.log
- adjoint-mixed: adj_fl16_training.log
- adjoint-mixed-bfloat: adj_bfl16_training.log
- direct: dir_training.log

Experiment Parameters:
- Network Architecture:
    - Same as Lars but with FDE blocks instead of ODE blocks

- FDE_Block:
    - Beta: 0.6
    - T: 1.0
    - step_size: 0.1
    - $f$ in $D^\beta z = f$: Time-dependent dynamics with piecewise-constant weights (same as Lars')

- Training Arguments:
    - Downsampling and other things exactly same as Lars
    - Epochs: 160 just for preliminary smoke tests
    - Batch Size: 16
    - Initial LR: 0.1
    - Momentum: 0.9
    - GPU: NVIDIA A100 (Palmetto)

Note: 
- adjoint mode uses adjoint method for gradients but in high precision
- adjoint-mixed mode uses adjoint method with float16 for mixed precision (and hence the DynamicScaler)
- adjoint-mixed-bflat uses adjoint method with bfloat16 for mixed precision (and hence no DynamicScaler)
- direct mode uses standard backprop with high precision
    
Training Plot (every 5 epochs):
![Training plots for peaks full experiment](./stl10_train_acc.png "Peaks full training curves")


