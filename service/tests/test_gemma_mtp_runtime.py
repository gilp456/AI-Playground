import os
import sys
import unittest
import logging
from unittest.mock import patch

os.environ.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:999")

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SERVICE_DIR)


class TestGemmaMtpRuntime(unittest.TestCase):
    def test_split_embedding_matches_scaled_embedding_forward(self):
        import torch
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding

        import gemma_mtp_runtime

        source = Gemma4TextScaledWordEmbedding(
            num_embeddings=8,
            embedding_dim=6,
            padding_idx=0,
            embed_scale=2.0,
        )
        with torch.no_grad():
            source.weight.copy_(torch.arange(48, dtype=torch.float32).reshape(8, 6))

        split = gemma_mtp_runtime._SplitScaledWordEmbedding.from_embedding(
            source,
            chunk_dim=2,
            device="cpu",
            dtype=torch.float32,
        )

        input_ids = torch.tensor([[1, 2, 3]])
        torch.testing.assert_close(split(input_ids), source(input_ids))
        self.assertEqual(len(split.chunks), 3)

    def test_split_embedding_state_dict_does_not_expose_original_large_weight(self):
        import torch
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding

        import gemma_mtp_runtime

        source = Gemma4TextScaledWordEmbedding(
            num_embeddings=4,
            embedding_dim=4,
            padding_idx=0,
            embed_scale=1.0,
        )

        split = gemma_mtp_runtime._SplitScaledWordEmbedding.from_embedding(
            source,
            chunk_dim=2,
            device="cpu",
            dtype=torch.float32,
        )

        self.assertNotIn("weight", split.state_dict())
        self.assertEqual({"chunks.0", "chunks.1"}, set(split.state_dict()))

    def test_gemma_mtp_load_endpoint_returns_backend_error_details(self):
        from web_api import app

        logging.disable(logging.CRITICAL)
        try:
            patcher = patch("gemma_mtp_runtime.load_model_pair", side_effect=RuntimeError("load failed"))
            with patcher:
                response = app.test_client().post(
                    "/api/gemmaMtp/load",
                    json={
                        "model": "google/gemma-4-E4B-it",
                        "assistant_model": "google/gemma-4-E4B-it-assistant",
                        "model_path": "C:/models",
                    },
                )
        finally:
            logging.disable(logging.NOTSET)

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["error"]["message"], "load failed")

    def test_gemma_mtp_unload_endpoint_releases_loaded_model(self):
        from web_api import app

        with patch("gemma_mtp_runtime.unload_model", return_value={"loaded": False}) as unload:
            response = app.test_client().post("/api/gemmaMtp/unload")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"loaded": False})
        unload.assert_called_once()


if __name__ == "__main__":
    unittest.main()
