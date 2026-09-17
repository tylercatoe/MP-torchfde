# Peaks Final Metrics Summary


```text
configuration                      | backward mode        | mesh    | solver              | precision | finalmse | best_mse | train_mem_mb | train_time_s | inf_time_s | inf_mem_mb
-----------------------------------+----------------------+---------+---------------------+-----------+----------+----------+--------------+--------------+------------+-----------
Direct Predictor · FP32            | direct AG            | uniform | predictor           | float32   | 1.4e-04  | 1.1e-04  | 1038.70      | 5589.46      | 0.0158     | 36.20       
Uniform Predictor · FP32           | adjoint              | uniform | predictor           | float32   | 2.1e-05  | 2.1e-05  | 542.15       | 5988.17      | 0.0107     | 47.43       
Uniform Predictor · FP16           | adjoint-mixed        | uniform | predictor           | float16   | 7.2e-05  | 6.8e-05  | 308.05       | 5326.97      | 0.0765     | 36.32       
Uniform Predictor · BF16           | adjoint-mixed-bfloat | uniform | predictor           | bfloat16  | 1.6e-04  | 1.4e-04  | 293.40       | 2616.10      | 0.0454     | 36.32       
Graded Predictor · FP32            | adjoint              | graded  | predictor           | float32   | 2.6e-05  | 2.6e-05  | 542.15       | 5692.53      | 0.0094     | 47.43       
Graded Predictor · FP16            | adjoint-mixed        | graded  | predictor           | float16   | 1.8e-04  | 1.2e-04  | 308.05       | 4125.77      | 0.0676     | 36.32       
Graded Predictor · BF16            | adjoint-mixed-bfloat | graded  | predictor           | bfloat16  | 1.7e-04  | 1.6e-04  | 293.40       | 2609.66      | 0.1092     | 36.32   
Uniform Predictor-Corrector · FP32 | adjoint              | uniform | predictor-corrector | float32   | 1.1e-04  | 1.1e-04  | 757.75       | 14401.02     | 0.0267     | 58.66       
Uniform Predictor-Corrector · FP16 | adjoint-mixed        | uniform | predictor-corrector | float16   | 1.5e-04  | 1.5e-04  | 410.59       | 11247.46     | 0.4996     | 41.93       
Uniform Predictor-Corrector · BF16 | adjoint-mixed-bfloat | uniform | predictor-corrector | bfloat16  | 3.6e-04  | 2.9e-04  | 401.58       | 6812.89      | 0.0605     | 41.93       
Graded Predictor-Corrector · FP32  | adjoint              | graded  | predictor-corrector | float32   | 2.5e-05  | 2.4e-05  | 757.75       | 14222.57     | 0.0269     | 58.66       
Graded Predictor-Corrector · FP16  | adjoint-mixed        | graded  | predictor-corrector | float16   | 2.5e-04  | 2.3e-04  | 410.59       | 12128.30     | 0.0855     | 41.93       
Graded Predictor-Corrector · BF16  | adjoint-mixed-bfloat | graded  | predictor-corrector | bfloat16  | 1.7e-04  | 1.7e-04  | 401.58       | 6465.99      | 0.0729     | 41.93       
```

Predictor: 
- Adjoint MP memory savings compared to direct AG: $71.8\\%$
- Adjoint MP memory savings compared to full precision adjoint: $45.9\\%$

Predictor-Corrector: 
- Adjoint MP memory savings compared to full precision adjoint: $47\\%$

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
    - Initial LR: 0.01
    - Weight decay: 5e-4
    - GPU: NVIDIA H200 (Palmetto)

Parameter count: 198,401

Note: 
- adjoint mode uses adjoint method for gradients but in high precision
- adjoint-mixed mode uses adjoint method with float16 for mixed precision (and hence the DynamicScaler)
- adjoint-mixed-bflat uses adjoint method with bfloat16 for mixed precision (and hence no DynamicScaler)
- direct mode uses standard backprop with high precision
    
Training Plots:
![Training plots for peaks full experiment](./peaks_train_mse_logscale.png "Peaks full training curves")

Testing Plots:
![Testing plots for peaks full experiment](./peaks_test_mse_logscale.png "Peaks full testing curves")


