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
    - Width: 256
    - Input layer -> tanh() -> FDE_Block -> Output layer 
    - Model parameter count: 198,401
- FDE_Block:
    - Beta: 0.5
    - T: 2.0
    - step_size: 0.1
    - $f$ in $D^\beta z = f$: 3 layer MLP
- Training Arguments:
    - Epochs: 5000
    - Batch size: 10,000
    - Total samples: 200,000
    - Initial LR: 0.1
    - Weight decay: 5e-4
    - GPU: NVIDIA H200 (Palmetto)

Note: 
- adjoint mode uses adjoint method for gradients but in high precision
- adjoint-mixed mode uses adjoint method with float16 for mixed precision (and hence the DynamicScaler)
- adjoint-mixed-bflat uses adjoint method with bfloat16 for mixed precision (and hence no DynamicScaler)
- direct mode uses standard backprop with high precision
    
Training Plot (every 50 epochs):
![Training plots for peaks full experiment](./peaks_train_mse_logscale.png "Peaks full training curves")


