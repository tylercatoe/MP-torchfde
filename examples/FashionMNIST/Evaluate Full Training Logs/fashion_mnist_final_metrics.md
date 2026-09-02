# Fashion MNIST Final Metrics Summary

## Full Training Metrics 

```text
mode                 | final_train_acc | final_val_err | best_val_err | train_mem_mb | train_time_s | infer_time_s | infer_mem_mb
---------------------+-----------------+---------------+--------------+--------------+--------------+--------------+-------------
adjoint              | 0.9996          | 0.0845        | 0.0745       | 171.27       | 1647.30      | 0.5900       | 120.24      
adjoint-mixed        | 0.9996          | 0.0773        | 0.0748       | 159.95       | 2021.81      | 0.7900       | 120.48      
adjoint-mixed-bfloat | 0.9996          | 0.0836        | 0.0766       | 159.95       | 2336.66      | 0.6700       | 120.48      
direct               | 0.9997          | 0.0798        | 0.0748       | 221.06       | 1280.29      | 0.7500       | 120.24      
```

Memory savings: $80.9\%$ between direct and adjoint MP (adjoint MP uses less)


Log files:
- adjoint: adj_full_logs.txt
- adjoint-mixed: adj_fl16_full_logs.txt
- adjoint-mixed-bfloat: adj_bfl16_full_logs.txt
- direct: dir_full_logs.txt


Experiment Parameters:
- Network Architecture:
    - Same as torchfde/Neural FDE paper (same as MNIST example)

- FDE_Block:
    - Beta: 0.3
    - T: 1.0
    - step_size: 0.1
    - $f$ in $D^\beta z = f$: Convolution Module

- Training Arguments:
    - Epochs: 160 
    - Batch Size: 128
    - Initial LR: 0.1, decay at specified boundary epochs 
    - Momentum: 0.9
    - Weight decay: 5e-4
    - GPU: NVIDIA H200 (Palmetto)

Parameter count: 208,266

Note: 
- adjoint mode uses adjoint method for gradients but in high precision
- adjoint-mixed mode uses adjoint method with float16 for mixed precision (and hence the DynamicScaler)
- adjoint-mixed-bflat uses adjoint method with bfloat16 for mixed precision (and hence no DynamicScaler)
- direct mode uses standard backprop with high precision
    
Training Plot (every epoch):
![Training plots for Fashion MNIST full experiment](./fashion_mnist_train_acc.png "Fashion MNIST full training curves")

Test Accuracy Plot (every epoch):
![Test accuracy plots for Fashion MNIST full experiment](./fashion_mnist_test_acc.png "Fashion MNIST full test curves")


## Fashion MNIST Final Time, T, Sweep Comparisions

We use the same network architecture and fractional dynamics as above, but now we sweep the final time $T$ across a range of values. Specifically, we take 
```math
T \in \{1, 2, 4, 8, 16, 32, 64, 128\}
```
and report peak GPU memory for each epoch and time per epoch for training. 

# Memory Results (Peak Memory, MB)
```text
|---------------|--------|--------|--------|--------|---------|---------|---------|----------|
|   method \ T  |   1    |   2    |   4    |   8    |   16    |   32    |   64    |   128    |
|---------------|--------|--------|--------|--------|---------|---------|---------|----------|
|      dir      | 221.06 | 312.35 | 494.94 | 860.12 | 1590.48 | 3051.29 | 5973.20 | 11818.20 |
|      adj      | 171.27 | 197.37 | 264.87 | 399.87 | 669.87  | 1209.87 | 2289.87 | 4449.87  |
|   adj_fl16    | 159.95 | 171.20 | 193.70 | 238.70 | 328.70  | 508.70  | 868.71  | 1588.71  |
|   adj_bfl16   | 159.95 | 171.20 | 193.70 | 238.70 | 328.70  | 508.70  | 868.71  | 1588.71  |
|---------------|--------|--------|--------|--------|---------|---------|---------|----------|
| dir/adj       | 1.291  | 1.583  | 1.869  | 2.151  | 2.374   | 2.522   | 2.609   | 2.656    |
| dir/adj_fl16  | 1.382  | 1.824  | 2.555  | 3.603  | 4.839   | 5.998   | 6.876   | 7.439    |
| dir/adj_bfl16 | 1.382  | 1.824  | 2.555  | 3.603  | 4.839   | 5.998   | 6.876   | 7.439    |
| adj/adj_fl16  | 1.071  | 1.153  | 1.367  | 1.675  | 2.038   | 2.378   | 2.636   | 2.801    |
| adj/adj_bfl16 | 1.071  | 1.153  | 1.367  | 1.675  | 2.038   | 2.378   | 2.636   | 2.801    |
|---------------|--------|--------|--------|--------|---------|---------|---------|----------|
```


# Time Results (s)
```text
|---------------|--------|--------|--------|--------|---------|---------|---------|----------|
|   method \ T  |   1    |   2    |   4    |   8    |   16    |   32    |   64    |   128    |
|---------------|--------|--------|--------|--------|---------|---------|---------|----------|
|      dir      | 6.44   | 9.20   | 23.97  | 63.99  | 202.36  | 795.04  | 3301.78 | 12404.80 |
|      adj      | 6.48   | 12.40  | 19.78  | 63.48  | 117.94  | 182.19  | 549.55  | 1334.01  |
|   adj_fl16    | 9.31   | 23.47  | 33.36  | 61.12  | 173.37  | 469.91  | 1149.10 | 4573.79  |
|   adj_bfl16   | 10.22  | 21.51  | 43.13  | 64.04  | 134.41  | 428.05  | 1108.14 | 3694.79  |
|---------------|--------|--------|--------|--------|---------|---------|---------|----------|
| dir/adj       | 0.994  | 0.742  | 1.212  | 1.008  | 1.716   | 4.364   | 6.008   | 9.299    |
| dir/adj_fl16  | 0.692  | 0.392  | 0.719  | 1.047  | 1.167   | 1.692   | 2.873   | 2.712    |
| dir/adj_bfl16 | 0.630  | 0.428  | 0.556  | 0.999  | 1.505   | 1.857   | 2.980   | 3.357    |
| adj/adj_fl16  | 0.696  | 0.529  | 0.593  | 1.039  | 0.680   | 0.388   | 0.478   | 0.292    |
| adj/adj_bfl16 | 0.634  | 0.577  | 0.459  | 0.991  | 0.877   | 0.426   | 0.496   | 0.361    |
|---------------|--------|--------|--------|--------|---------|---------|---------|----------|
```
