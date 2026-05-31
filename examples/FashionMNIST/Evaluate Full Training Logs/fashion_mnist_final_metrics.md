# Fashion MNIST Final Metrics Summary

## Full Training Metrics 

```text
mode                 | final_val_err  | best_val_err | train_mem_mb | train_time_s | infer_time_s | infer_mem_mb |
---------------------+----------------+--------------+--------------+--------------+--------------+--------------+
adjoint              | 0.0785         | 0.0736       | 171.27       | 1658.22      | 0.6300       | 1831.83      |
adjoint-mixed        | 0.0838         | 0.0764       | 164.32       | 3095.77      | 0.7000       | 1698.25      |
adjoint-mixed-bfloat | 0.0806         | 0.076        | 163.20       | 1856.93      | 0.6700       | 1698.25      |
direct               | 0.083          | 0.0764       | 221.06       | 1231.05      | 0.6500       | 352.59       |
```

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
    - GPU: NVIDIA H200 (Palmetto)

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
|   adj_fl16    | 164.32 | 185.98 | 230.98 | 320.98 | 500.98  | 860.98  | 1580.98 | 3020.99  |
|   adj_bfl16   | 163.20 | 185.70 | 230.70 | 320.70 | 500.70  | 860.70  | 1580.70 | 3020.71  |
|---------------|--------|--------|--------|--------|---------|---------|---------|----------|
| dir/adj_fl16  | 1.345  | 1.680  | 2.143  | 2.680  | 3.175   | 3.544   | 3.778   | 3.912    |
| dir/adj_bfl16 | 1.355  | 1.682  | 2.145  | 2.682  | 3.177   | 3.545   | 3.779   | 3.912    |
| adj/adj_fl16  | 1.042  | 1.061  | 1.147  | 1.246  | 1.337   | 1.405   | 1.448   | 1.473    |
| adj/adj_bfl16 | 1.049  | 1.063  | 1.148  | 1.247  | 1.338   | 1.406   | 1.449   | 1.473    |
|---------------|--------|--------|--------|--------|---------|---------|---------|----------|
```

# Time Results (s)
```text
|---------------|--------|--------|--------|--------|---------|---------|---------|----------|
|   method \ T  |   1    |   2    |   4    |   8    |   16    |   32    |   64    |   128    |
|---------------|--------|--------|--------|--------|---------|---------|---------|----------|
|      dir      | 4.76   | 8.97   | 21.29  | 60.54  | 197.14  | 771.63  | 3050.75 | 13069.61 |
|      adj      | 7.27   | 12.79  | 23.79  | 45.40  | 88.51   | 182.66  | 432.50  | 1271.67  | 
|   adj_fl16    | 13.65  | 24.49  | 57.29  | 110.40 | 244.88  | 479.82  | 1103.12 | 2790.09  |
|   adj_bfl16   | 8.24   | 14.84  | 27.41  | 52.84  | 102.76  | 215.49  | 515.83  | 1524.71  |
|---------------|--------|--------|--------|--------|---------|---------|---------|----------|
| dir/adj_fl16  | 0.349  | 0.366  | 0.372  | 0.548  | 0.805   | 1.608   | 2.766   | 4.684    |
| dir/adj_bfl16 | 0.577  | 0.604  | 0.777  | 1.146  | 1.919   | 3.581   | 5.914   | 8.572    |
| adj/adj_fl16  | 0.532  | 0.522  | 0.415  | 0.411  | 0.361   | 0.381   | 0.392   | 0.456    |
| adj/adj_bfl16 | 0.882  | 0.862  | 0.868  | 0.859  | 0.861   | 0.848   | 0.838   | 0.834    |
|---------------|--------|--------|--------|--------|---------|---------|---------|----------|
```