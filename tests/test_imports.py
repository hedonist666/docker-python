import unittest

class TestImport(unittest.TestCase):
    # Basic import tests for packages without any.
    def test_basic(self):
        import bq_helper
        import tensorflow_datasets
        import segment_anything
