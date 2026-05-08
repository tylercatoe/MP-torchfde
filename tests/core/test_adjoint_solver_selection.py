import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from torchfde.fdeadjoint import (
    DynamicScaler,
    FDEAdjointMethodDynamic,
    FDEAdjointMethodUnscaled,
    FDEAdjointMethodUnscaledSafe,
    _select_adjoint_solver,
)


class TestAdjointSolverSelection(unittest.TestCase):
    def test_selection_matrix(self):
        cases = [
            (None, torch.float64, FDEAdjointMethodUnscaled, None),
            (None, torch.float32, FDEAdjointMethodUnscaled, None),
            (None, torch.bfloat16, FDEAdjointMethodUnscaled, None),
            (None, torch.float16, FDEAdjointMethodDynamic, DynamicScaler),
            (False, torch.float16, FDEAdjointMethodUnscaledSafe, None),
            (False, torch.float32, FDEAdjointMethodUnscaled, None),
            (DynamicScaler(torch.float16), torch.float16, FDEAdjointMethodDynamic, DynamicScaler),
        ]

        for loss_scaler, precision, expected_solver, expected_scaler_type in cases:
            with self.subTest(loss_scaler=type(loss_scaler).__name__ if loss_scaler is not None else "None",
                              precision=str(precision)):
                solver, scaler = _select_adjoint_solver(loss_scaler, precision)
                self.assertEqual(solver, expected_solver)
                if expected_scaler_type is None:
                    self.assertIsNone(scaler)
                else:
                    self.assertIsInstance(scaler, expected_scaler_type)


if __name__ == "__main__":
    unittest.main()
