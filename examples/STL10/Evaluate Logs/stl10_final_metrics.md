# STL10 Final Metrics Summary

## Full Training Metrics

```text
configuration            | backward mode        | mesh    | precision | final_acc | best_acc | train_mem_mb | train_time_s | inf_time_s | inf_mem_mb
-------------------------+----------------------+---------+-----------+-----------+----------+--------------+--------------+------------+-----------
Direct Predictor · FP32  | direct AG            | uniform | float32   | 0.6890    | 0.6890   | 4299.64      | 4047.21      | 1.72       | 1130.14
Uniform Predictor · FP32 | adjoint              | uniform | float32   | 0.7050    | 0.7120   | 2296.38      | 4553.96      | 1.50       | 1962.14
Uniform Predictor · FP16 | adjoint-mixed        | uniform | float16   | 0.6890    | 0.6960   | 1209.97      | 3330.27      | 1.11       | 1034.23
Uniform Predictor · BF16 | adjoint-mixed-bfloat | uniform | bfloat16  | 0.7010    | 0.7090   | 1207.71      | 3122.13      | 1.47       | 1034.23
```

Predictor:
- Adjoint MP memory savings compared to direct AG: $71.9\%$
- Adjoint MP memory savings compared to full precision adjoint: $47.4\%$

Predictor-Corrector:
- Adjoint MP memory savings compared to full precision adjoint: $N/A$

Experiment Parameters:
- Network Architecture:
    - STL10 convolutional Neural FDE classifier
    - Width: 128
    - Model parameter count: 3,144,970
- FDE_Block:
    - Beta: 0.6
    - T: 1.0
    - step_size: 0.1
    - $f$ in $D^\beta z = f$: time-dependent dynamics with piecewise-constant weights
- Training Arguments:
    - Epochs: 160
    - Batch size: 16
    - Initial LR: 0.05, decayed by the training schedule
    - Momentum: 0.9
    - Weight decay: 5e-4
    - GPU: NVIDIA H200 (Palmetto)

Parameter count: 3,144,970

Note:
- These historical metrics contain only the uniform predictor configurations; the updated launcher generates the complete 13-run table.
- adjoint mode uses the custom adjoint in float32 throughout
- adjoint-mixed mode uses float16 adjoint storage and the DynamicScaler
- adjoint-mixed-bfloat uses bfloat16 adjoint storage without dynamic scaling
- direct mode uses standard backpropagation in float32
- graded and uniform specify the shared forward/backward time mesh

Training Plot (every logged epoch):
![Training plot for STL10](./stl10_train_acc.png "STL10 training curves")

Validation Accuracy Plot (every logged epoch):
![Validation plot for STL10](./stl10_test_acc.png "STL10 validation curves")
