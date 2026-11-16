import unittest

import omnisearch_ai


class PackageSmokeTest(unittest.TestCase):
    def test_package_has_initial_version(self) -> None:
        self.assertEqual(omnisearch_ai.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
