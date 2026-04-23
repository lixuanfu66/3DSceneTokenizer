from __future__ import annotations

import unittest

from threedvae.utils.torch_compat import HAS_TORCH, require_torch


class TorchGuardTest(unittest.TestCase):
    def test_require_torch_behavior(self) -> None:
        if HAS_TORCH:
            modules = require_torch()
            self.assertEqual(len(modules), 5)
        else:
            with self.assertRaises(RuntimeError):
                require_torch()
