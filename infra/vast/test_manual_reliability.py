"""Control-plane regressions observed in the manual session; no network/model."""
import pathlib
import tempfile
import unittest
from unittest.mock import patch
import controller
import watchdog
import probe
import json
import subprocess


class ManualReliability(unittest.TestCase):
    def test_single_watchdog_across_worktrees(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(pathlib.Path(d) / "program.lock")
            first = watchdog.acquire_watchdog_lock(path)
            try:
                with self.assertRaises(SystemExit):
                    watchdog.acquire_watchdog_lock(path)
            finally:
                first.close()
            watchdog.acquire_watchdog_lock(path).close()

    def test_status_omits_credentials(self):
        program = json.loads((probe.HERE / "program.json").read_text())["program"]
        raw = json.dumps([{"id": 123, "label": program + "_test", "extra_env": {"TOKEN": "synthetic-secret"},
                           "jupyter_token": "synthetic-secret", "onstart": "synthetic-secret"}])
        with patch.object(probe.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, raw, "")):
            out = probe.status("test")
        self.assertEqual(out[0]["id"], 123)
        self.assertNotIn("synthetic-secret", json.dumps(out))

    def test_env_preserves_gpu_name_and_rejects_unsupported_quotes(self):
        self.assertEqual(controller.format_env({"EXPECTED_GPU": "RTX PRO 5000", "EMPTY": "", "ZERO": 0}),
                         '-e EXPECTED_GPU="RTX PRO 5000" -e ZERO="0"')
        with self.assertRaises(AssertionError):
            controller.format_env({"TOKEN": 'unsafe" -e OTHER=value'})

    def test_retry_gpu_and_total_cost(self):
        offers = [dict(id=1, gpu_name="L40S", dph_total=.6, inet_down_cost=.04),
                  dict(id=2, gpu_name="RTX PRO 5000", dph_total=.5, inet_down_cost=0),
                  dict(id=3, gpu_name="L40S", dph_total=.8, inet_down_cost=0)]
        man = dict(gpu_query="gpu_ram>=40", disk=80, expected_gpu="L40S",
                   estimated_billed_h=1, estimated_download_gb=32)
        with patch.object(controller, "vast", return_value=offers), patch.object(controller, "wedged_machines", return_value=set()):
            self.assertEqual([o["id"] for o in controller.search_offers(man)], [3, 1])


if __name__ == "__main__":
    unittest.main()
