# Lotka--Volterra Final Metrics Summary

## Full Training Metrics

```text
configuration                                    | backward mode | final_train | final_val  | best_val   | best_iter | param_err  | peak_mem_mb | mean_iter_s | est_train_s
-------------------------------------------------+---------------+-------------+------------+------------+-----------+------------+-------------+-------------+------------
near_true · predictor · uniform · FP32           | adjoint       | 0.00236872  | 0.00320365 | 0.00316636 | 50        | 0.00626267 | 0.05888     | 0.0669767   | 33.4884
near_true · predictor · uniform · FP16           | adjoint-mixed | 0.00236866  | 0.00320413 | 0.00316658 | 50        | 0.00625885 | 0.038912    | 0.0822018   | 41.1009
near_true · predictor · graded · FP32            | adjoint       | 0.00241489  | 0.00324488 | 0.00322146 | 75        | 0.00716326 | 0.058368    | 0.0541925   | 27.0963
near_true · predictor · graded · FP16            | adjoint-mixed | 0.00241472  | 0.0032449  | 0.00322112 | 75        | 0.00715863 | 0.0384      | 0.0808647   | 40.4323
near_true · predictor-corrector · uniform · FP32 | adjoint       | 0.00245521  | 0.00333302 | 0.00333301 | 275       | 0.00217439 | 0.081408    | 0.177432    | 88.7159
near_true · predictor-corrector · uniform · FP16 | adjoint-mixed | 0.00245497  | 0.00333343 | 0.00333343 | 500       | 0.00227007 | 0.051712    | 0.242598    | 121.299
near_true · predictor-corrector · graded · FP32  | adjoint       | 0.00248477  | 0.00343889 | 0.00343889 | 275       | 0.00354832 | 0.081408    | 0.176951    | 88.4756
near_true · predictor-corrector · graded · FP16  | adjoint-mixed | 0.0024849   | 0.00343882 | 0.0034386  | 375       | 0.00352834 | 0.051712    | 0.244847    | 122.423
-------------------------------------------------+---------------+-------------+------------+------------+-----------+------------+-------------+-------------+------------
worse · predictor · uniform · FP32               | adjoint       | 0.002385    | 0.00320556 | 0.00320556 | 500       | 0.0121271  | 0.05888     | 0.054542    | 27.271 
worse · predictor · uniform · FP16               | adjoint-mixed | 0.00238534  | 0.00320506 | 0.00320506 | 500       | 0.0121291  | 0.038912    | 0.0814781   | 40.7391
worse · predictor · graded · FP32                | adjoint       | 0.00243241  | 0.00323914 | 0.00323914 | 500       | 0.0118879  | 0.058368    | 0.0538821   | 26.941 
worse · predictor · graded · FP16                | adjoint-mixed | 0.00243263  | 0.00323973 | 0.00323973 | 500       | 0.0118859  | 0.0384      | 0.0812588   | 40.6294
worse · predictor-corrector · uniform · FP32     | adjoint       | 0.00246796  | 0.00327887 | 0.00327193 | 450       | 0.00619064 | 0.081408    | 0.177851    | 88.9257
worse · predictor-corrector · uniform · FP16     | adjoint-mixed | 0.00246744  | 0.00327892 | 0.00327223 | 450       | 0.00616879 | 0.051712    | 0.243615    | 121.808
worse · predictor-corrector · graded · FP32      | adjoint       | 0.00249719  | 0.0033644  | 0.00334563 | 425       | 0.0072458  | 0.081408    | 0.176737    | 88.3684
worse · predictor-corrector · graded · FP16      | adjoint-mixed | 0.00249673  | 0.00336486 | 0.00334527 | 425       | 0.00713947 | 0.051712    | 0.241991    | 120.996
```

FP16 memory savings compared with FP32:
- near_true, predictor, uniform: $33.9\\%$
- near_true, predictor, graded: $34.2\\%$
- near_true, predictor-corrector, uniform: $36.5\\%$
- near_true, predictor-corrector, graded: $36.5\\%$
- worse, predictor, uniform: $33.9\\%$
- worse, predictor, graded: $34.2\\%$
- worse, predictor-corrector, uniform: $36.5\\%$
- worse, predictor-corrector, graded: $36.5\\%$

Experiment Parameters:
- Fractional Lotka--Volterra system:
    - $D^\beta x = x(a-cy)$
    - $D^\beta y = -y(b-dx)$
    - True parameters $[a,b,c,d]$: [1.0, 0.5, 1.0, 0.3]
    - Beta: 0.7
    - T: 5.0
    - Training step size: 0.1
    - Synthetic-data step size: 0.02
- Training Arguments:
    - Iterations: 500
    - Training trajectories: 50
    - Validation trajectories: 25
    - Noise standard deviation: 0.05
    - Learning rate: 0.01
    - Seed: 42

```text
Initialization and Final Learned Parameters:
- near_true · predictor · uniform · FP32:             initial [0.990000, 0.480000, 1.050000, 0.330000]     final [1.002521, 0.487760, 0.999159, 0.290551]
- near_true · predictor · uniform · FP16:             initial [0.990000, 0.480000, 1.050000, 0.330000]     final [1.002488, 0.487767, 0.999131, 0.290555]
- near_true · predictor · graded · FP32:              initial [0.990000, 0.480000, 1.050000, 0.330000]     final [1.004222, 0.486438, 1.001205, 0.290336]
- near_true · predictor · graded · FP16:              initial [0.990000, 0.480000, 1.050000, 0.330000]     final [1.004216, 0.486443, 1.001200, 0.290339]
- near_true · predictor-corrector · uniform · FP32:   initial [0.990000, 0.480000, 1.050000, 0.330000]     final [1.003238, 0.497961, 1.002203, 0.298782]
- near_true · predictor-corrector · uniform · FP16:   initial [0.990000, 0.480000, 1.050000, 0.330000]     final [1.003404, 0.497927, 1.002359, 0.298756]
- near_true · predictor-corrector · graded · FP32:    initial [0.990000, 0.480000, 1.050000, 0.330000]     final [1.003972, 0.502827, 1.005322, 0.302073]
- near_true · predictor-corrector · graded · FP16:    initial [0.990000, 0.480000, 1.050000, 0.330000]     final [1.003977, 0.502779, 1.005334, 0.302023]
- worse · predictor · uniform · FP32:                 initial [0.650000, 0.750000, 1.350000, 0.180000]     final [0.985104, 0.490537, 0.983787, 0.292063]
- worse · predictor · uniform · FP16:                 initial [0.650000, 0.750000, 1.350000, 0.180000]     final [0.985102, 0.490536, 0.983782, 0.292063]
- worse · predictor · graded · FP32:                  initial [0.650000, 0.750000, 1.350000, 0.180000]     final [0.986135, 0.489260, 0.985202, 0.291851]
- worse · predictor · graded · FP16:                  initial [0.650000, 0.750000, 1.350000, 0.180000]     final [0.986141, 0.489262, 0.985202, 0.291851]
- worse · predictor-corrector · uniform · FP32:       initial [0.650000, 0.750000, 1.350000, 0.180000]     final [0.987203, 0.500349, 0.988407, 0.300024]
- worse · predictor-corrector · uniform · FP16:       initial [0.650000, 0.750000, 1.350000, 0.180000]     final [0.987226, 0.500331, 0.988441, 0.300011]
- worse · predictor-corrector · graded · FP32:        initial [0.650000, 0.750000, 1.350000, 0.180000]     final [0.987670, 0.504922, 0.991296, 0.303027]
- worse · predictor-corrector · graded · FP16:        initial [0.650000, 0.750000, 1.350000, 0.180000]     final [0.987877, 0.504900, 0.991480, 0.303015]
```

Note:
- All configurations use the same seeded train/validation data and initialization within each initialization regime.
- Predictor and predictor-corrector are tested independently on uniform and double-graded meshes.
- FP16 uses the safe mixed-precision custom adjoint; FP32 is the full-precision custom adjoint.
- `param_err` is the final mean absolute error in the four learned parameters.
- `est_train_s` extrapolates mean measured iteration time across all training iterations.

<!-- Validation Loss versus Iteration:
![Validation loss versus iteration](./validation_loss_vs_iteration.png) 

Validation Loss versus Estimated Time:
![Validation loss versus time](./validation_loss_vs_time.png) 
-->

Parameter Trajectories:
![Parameter trajectories](./parameter_trajectories.png)

Relative Parameter Error:
![Relative parameter error](./relative_parameter_error_vs_iteration.png)

<!-- Accuracy--Cost Tradeoff:
![Accuracy cost tradeoff](./accuracy_cost_tradeoff.png)
-->
