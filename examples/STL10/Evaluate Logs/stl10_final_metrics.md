# STL10 Final Metrics Summary

## Full Training Metrics

```text
configuration                      | backward mode        | mesh    | precision | status | final_acc | best_acc | train_mem_mb | train_time_s | inf_time_s | inf_mem_mb
-----------------------------------+----------------------+---------+-----------+--------+-----------+----------+--------------+--------------+------------+-----------
Direct Predictor · FP32            | direct AG            | uniform | float32   | ok     | 0.7420    | 0.7470   | 8580.14      | 7761.33      | 3.17       | 2262.36   
Uniform Predictor · FP32           | adjoint              | uniform | float32   | ok     | 0.7430    | 0.7530   | 4613.38      | 8501.27      | 3.00       | 3926.37   
Uniform Predictor · FP16           | adjoint-mixed        | uniform | float16   | ok     | 0.7290    | 0.7320   | 2595.53      | 10173.68     | 2.22       | 2070.70   
Uniform Predictor · BF16           | adjoint-mixed-bfloat | uniform | bfloat16  | ok     | 0.7310    | 0.7350   | 2430.95      | 5676.79      | 2.07       | 2070.70   
Graded Predictor · FP32            | adjoint              | graded  | float32   | ok     | 0.7860    | 0.7920   | 4613.38      | 8877.01      | 3.07       | 3926.37   
Graded Predictor · FP16            | adjoint-mixed        | graded  | float16   | ok     | 0.7730    | 0.7760   | 2595.53      | 8652.69      | 2.06       | 2070.70   
Graded Predictor · BF16            | adjoint-mixed-bfloat | graded  | bfloat16  | ok     | 0.7670    | 0.7740   | 2430.95      | 5662.08      | 2.07       | 2070.70   
Uniform Predictor-Corrector · FP32 | adjoint              | uniform | float32   | ok     | 0.0990    | 0.5130   | 7145.39      | 18963.16     | 6.80       | 5590.37   
Uniform Predictor-Corrector · FP16 | adjoint-mixed        | uniform | float16   | FAIL   | F         | F        | F            | F            | F          | F         
Uniform Predictor-Corrector · BF16 | adjoint-mixed-bfloat | uniform | bfloat16  | ok     | 0.0990    | 0.4350   | 3714.96      | 12118.84     | 4.56       | 2902.70   
Graded Predictor-Corrector · FP32  | adjoint              | graded  | float32   | ok     | 0.6550    | 0.6620   | 7145.39      | 18256.78     | 6.67       | 5590.37   
Graded Predictor-Corrector · FP16  | adjoint-mixed        | graded  | float16   | ok     | 0.6240    | 0.6240   | 3760.91      | 20616.05     | 4.92       | 2902.70   
Graded Predictor-Corrector · BF16  | adjoint-mixed-bfloat | graded  | bfloat16  | ok     | 0.6570    | 0.6620   | 3714.96      | 12281.40     | 4.61       | 2902.70   
```

Predictor:
- Adjoint MP memory savings compared to direct AG: $71.7\%$
- Adjoint MP memory savings compared to full precision adjoint: $47.3\%$

Predictor-Corrector:
- Adjoint MP memory savings compared to full precision adjoint: $48.0\%$

Experiment Parameters:
- Network Architecture:
    - STL10 convolutional Neural FDE classifier
    - Width: 128
    - Model parameter count: 12,565,002
- FDE_Block:
    - Beta: 0.6
    - T: 1.0
    - step_size: 0.1
    - $f$ in $D^\beta z = f$: Convolution module
- Training Arguments:
    - Epochs: 160
    - Batch size: 16
    - Initial LR: 0.05, decayed by the training schedule
    - Momentum: 0.9
    - Weight decay: 5e-4
    - GPU: NVIDIA H200 (Palmetto)

Parameter count: 12,565,002

Note:
- adjoint mode uses the custom adjoint in float32 throughout
- adjoint-mixed mode uses float16 adjoint storage and the DynamicScaler
- adjoint-mixed-bfloat uses bfloat16 adjoint storage without dynamic scaling
- direct mode uses standard backpropagation in float32
- graded and uniform specify the shared forward/backward time mesh

Failed Configurations:
- Uniform Predictor-Corrector · FP16: training log has no Final metrics line

Training Plot (every logged epoch):
![Training plot for STL10](./stl10_train_acc.png "STL10 training curves")

Validation Accuracy Plot (every logged epoch):
![Validation plot for STL10](./stl10_test_acc.png "STL10 validation curves")
