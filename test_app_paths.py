import os
import tempfile
import unittest
from pathlib import Path

from app_paths import APP_ROOT, asset_path


class AssetPathTests(unittest.TestCase):
    def test_assets_resolve_from_application_root_outside_repo(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                self.assertEqual(
                    asset_path("sentiment_model.keras"),
                    APP_ROOT / "sentiment_model.keras",
                )
            finally:
                os.chdir(original)

    def test_asset_path_rejects_parent_traversal(self):
        with self.assertRaises(ValueError):
            asset_path("../sentiment_model.keras")


if __name__ == "__main__":
    unittest.main()
