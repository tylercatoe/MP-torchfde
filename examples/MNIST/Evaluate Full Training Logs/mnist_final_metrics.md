# MNIST Final Metrics Summary

## Full Training Metrics

```text
configuration            | backward mode        | mesh    | precision | final_acc | best_acc | train_mem_mb | train_time_s | inf_time_s | inf_mem_mb
-------------------------+----------------------+---------+-----------+-----------+----------+--------------+--------------+------------+-----------
Direct Predictor · FP32  | direct AG            | uniform | float32   | 0.9937    | 0.9946   | 1955.68      | 27384.28     | 20.83      | 299.19
Uniform Predictor · FP32 | adjoint              | uniform | float32   | 0.9942    | 0.9952   | 804.87       | 10576.97     | 5.33       | 750.06
Uniform Predictor · FP16 | adjoint-mixed        | uniform | float16   | 0.9937    | 0.9943   | 373.70       | 19381.30     | 12.87      | 300.65
Uniform Predictor · BF16 | adjoint-mixed-bfloat | uniform | bfloat16  | 0.9940    | 0.9943   | 373.70       | 20987.90     | 13.53      | 300.65
```

Predictor:
- Adjoint MP memory savings compared to direct AG: $80.9\%$
- Adjoint MP memory savings compared to full precision adjoint: $53.6\%$

Predictor-Corrector:
- Adjoint MP memory savings compared to full precision adjoint: $N/A$

Experiment Parameters:
- Network Architecture:
    - Same as the torchfde Neural FDE paper
    - Model parameter count: 208,266
- FDE_Block:
    - Beta: 0.5
    - T: 20.0
    - step_size: 0.1
    - $f$ in $D^\beta z = f$: Convolution module
- Training Arguments:
    - Epochs: 60
    - Batch size: 128
    - Initial LR: 0.1, decayed at the specified boundary epochs
    - Momentum: 0.9
    - Weight decay: 5e-4
    - GPU: NVIDIA H200 (Palmetto)

Parameter count: 208,266

Note:
- These historical metrics contain only the uniform predictor configurations; the updated launcher generates the complete 13-run table.
- adjoint mode uses the custom adjoint in float32 throughout
- adjoint-mixed mode uses float16 adjoint storage and the DynamicScaler
- adjoint-mixed-bfloat uses bfloat16 adjoint storage without dynamic scaling
- direct mode uses standard backpropagation in float32
- graded and uniform specify the shared forward/backward time mesh

Training Plot (every logged epoch):
![Training plot for MNIST](./mnist_train_acc.png "MNIST training curves")

Validation Accuracy Plot (every logged epoch):
![Validation plot for MNIST](./mnist_test_acc.png "MNIST validation curves")
