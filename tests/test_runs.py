import json
import tempfile
import unittest
from pathlib import Path

from pvz_deeplearning.runs import RunDirectory, RunManifest, checkpoint_sha256, validate_resume
from pvz_deeplearning.harness import EXPECTED_HARNESS_CONTRACT


def manifest():
    return RunManifest(1, "run", "now", "v", "sha", False, "v0.1.0", "hsha", "game",
        EXPECTED_HARNESS_CONTRACT, {"id": "level"}, "maskable_ppo", "sb3-contrib", "2.9.0",
        {"id": "maskable_ppo_mlp_small"}, {}, {"python": 1}, False, .25, 1., {"total": 1},
        None, None, "3.12", "2", None, "cpu", None, {})


class RunTests(unittest.TestCase):
    def test_manifest_write_once_and_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            run = RunDirectory(folder, manifest()); path = run.create({"x": 1})
            self.assertTrue((path / "checkpoints").is_dir())
            with self.assertRaises(FileExistsError): run.write_once("manifest.json", {})

    def test_checkpoint_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x"; path.write_bytes(b"abc")
            self.assertEqual(checkpoint_sha256(path), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_completion_record_is_immutable_and_hashes_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            run = RunDirectory(folder, manifest()); run.create({"x": 1})
            checkpoint = run.path / "checkpoints" / "latest.zip"
            checkpoint.write_bytes(b"abc")
            completion = run.complete(checkpoint, final_step=12, completion_reason="total_timesteps")
            data = json.loads(completion.read_text(encoding="utf-8"))
            self.assertEqual(data["final_step"], 12)
            self.assertEqual(data["checkpoint_sha256"], checkpoint_sha256(checkpoint))
            with self.assertRaises(FileExistsError):
                run.complete(checkpoint, final_step=13, completion_reason="overwrite")

    def test_resume_mismatch_rejected(self):
        data = manifest().to_dict(); data["algorithm"] = "other"
        with self.assertRaises(ValueError): validate_resume(data, algorithm="maskable_ppo", model_id="maskable_ppo_mlp_small")
