# MNIST Final Metrics Summary

## Full Training Metrics 

```text
mode                 | final_train_acc | final_val_err | best_val_err | train_mem_mb | train_time_s | infer_time_s | infer_mem_mb
---------------------+-----------------+---------------+--------------+--------------+--------------+--------------+-------------
adjoint              | 0.9999          | 0.0058        | 0.0048       | 804.87       | 10576.97     | 5.33         | 750.06   
adjoint-mixed        | 0.9998          | 0.0063        | 0.0057       | 373.70       | 19381.30     | 12.87        | 300.65 
adjoint-mixed-bfloat | 0.9998          | 0.0060        | 0.0057       | 373.70       | 20987.90     | 13.53        | 300.65      
direct               | 0.9998          | 0.0063        | 0.0054       | 1955.68      | 27384.28     | 20.83        | 299.19     
```

Memory savings: $80.9\%$ between direct and adjoint MP (adjoint MP uses less)

Log files:
- adjoint: adj_full_logs.txt
- adjoint-mixed: adj_fl16_full_logs.txt
- adjoint-mixed-bfloat: adj_bfl16_full_logs.txt
- direct: dir_full_logs.txt


Experiment Parameters:
- Network Architecture:
    - Same as torchfde/Neural FDE paper 

- FDE_Block:
    - Beta: 0.5
    - T: 20.0
    - step_size: 0.1
    - $f$ in $D^\beta z = f$: Convolution Module

- Training Arguments:
    - Epochs: 60 
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
![Training plots for MNIST full experiment](./mnist_train_acc.png "MNIST full training curves")

Test Accuracy Plot (every epoch):
![Test accuracy plots for MNIST full experiment](./mnist_test_acc.png "MNIST full test curves")


