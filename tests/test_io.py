import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from numcompute import read_csv, read_csv_chunks


class TestReadCSV(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            suffix=".csv"
        )

        self.temp_file.write(
            "height,weight\n"
            "170,65\n"
            "180,80\n"
            "175,70\n"
        )

        self.temp_file.close()

        self.filepath = self.temp_file.name

    def tearDown(self):
        """Remove the temporary CSV file after each test."""
        os.remove(self.filepath)

    def test_basic_csv_reading(self):
        """Test basic CSV loading."""
        data = read_csv(
            self.filepath,
            delimiter=",",
            skip_header=1
        )

        expected = np.array([
            [170, 65],
            [180, 80],
            [175, 70]
        ], dtype=float)

        np.testing.assert_array_equal(data, expected)

    def test_returns_numpy_array(self):
        """Test that read_csv returns a NumPy array."""
        data = read_csv(
            self.filepath,
            delimiter=",",
            skip_header=1
        )

        self.assertIsInstance(data, np.ndarray)

    def test_custom_delimiter(self):
        """Test reading a tab-separated file."""

        with tempfile.NamedTemporaryFile(
                mode="w",
                delete=False,
                suffix=".tsv"
        ) as f:
            f.write(
                "height\tweight\n"
                "170\t65\n"
                "180\t80\n"
            )
            filepath = f.name

        try:
            data = read_csv(
                filepath,
                delimiter="\t",
                skip_header=1
            )

            expected = np.array([
                [170, 65],
                [180, 80]
            ], dtype=float)

            np.testing.assert_array_equal(data, expected)

        finally:
            os.remove(filepath)

    def test_missing_values(self):
        """Test handling of missing values."""

        with tempfile.NamedTemporaryFile(
                mode="w",
                delete=False,
                suffix=".csv"
        ) as f:
            f.write(
                "height,weight\n"
                "170,65\n"
                "180,\n"
                ",70\n"
            )
            filepath = f.name

        try:
            data = read_csv(
                filepath,
                delimiter=",",
                missing_values=[""],
                dtype=float,
                filling_values=np.nan,
                skip_header=1
            )

            self.assertTrue(np.isnan(data[1, 1]))
            self.assertTrue(np.isnan(data[2, 0]))

        finally:
            os.remove(filepath)

    def test_custom_filling_value(self):
        """Test replacing missing values with a custom value."""

        with tempfile.NamedTemporaryFile(
                mode="w",
                delete=False,
                suffix=".csv"
        ) as f:
            f.write(
                "height,weight\n"
                "170,65\n"
                "180,\n"
                ",70\n"
            )
            filepath = f.name

        try:
            data = read_csv(
                filepath,
                delimiter=",",
                missing_values=[""],
                filling_values=-1,
                skip_header=1
            )

            self.assertEqual(data[1, 1], -1)
            self.assertEqual(data[2, 0], -1)

        finally:
            os.remove(filepath)

    def test_dtype_float(self):
        """Test explicit float dtype."""

        data = read_csv(
            self.filepath,
            delimiter=",",
            dtype=float,
            skip_header=1
        )

        self.assertEqual(data.dtype, np.float64)

    def test_skip_header(self):
        """Test skipping the header row."""

        data = read_csv(
            self.filepath,
            delimiter=",",
            skip_header=1
        )

        self.assertEqual(data.shape, (3, 2))

    def test_single_row(self):
        """Test CSV containing a single data row."""

        with tempfile.NamedTemporaryFile(
                mode="w",
                delete=False,
                suffix=".csv"
        ) as f:
            f.write(
                "height,weight\n"
                "170,65\n"
            )
            filepath = f.name

        try:
            data = read_csv(
                filepath,
                delimiter=",",
                skip_header=1
            )

            self.assertEqual(data.shape, (1, 2))

        finally:
            os.remove(filepath)

    def test_single_column(self):
        """Test CSV containing a single column."""

        with tempfile.NamedTemporaryFile(
                mode="w",
                delete=False,
                suffix=".csv"
        ) as f:
            f.write(
                "height\n"
                "170\n"
                "180\n"
                "175\n"
            )
            filepath = f.name

        try:
            data = read_csv(
                filepath,
                delimiter=",",
                skip_header=1
            )

            self.assertEqual(data.shape, (3, 1))

        finally:
            os.remove(filepath)

    def test_single_value(self):
        """Test CSV containing a single value."""

        with tempfile.NamedTemporaryFile(
                mode="w",
                delete=False,
                suffix=".csv"
        ) as f:
            f.write(
                "height\n"
                "170\n"
            )
            filepath = f.name

        try:
            data = read_csv(
                filepath,
                delimiter=",",
                skip_header=1
            )

            self.assertEqual(data.shape, (1, 1))
            self.assertEqual(data[0, 0], 170)

        finally:
            os.remove(filepath)

    def test_missing_file(self):
        """Test that a nonexistent file raises ValueError."""

        with self.assertRaises(ValueError):
            read_csv("data.csv")


class TestReadCSVChunks(unittest.TestCase):

    def setUp(self):
        """Create a temporary CSV file."""

        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            suffix=".csv"
        )

        self.temp_file.write(
            "height,weight\n"
            "170,65\n"
            "180,80\n"
            "175,70\n"
            "165,60\n"
            "172,68\n"
        )

        self.temp_file.close()

        self.filepath = self.temp_file.name

    def tearDown(self):
        """Remove temporary file."""
        os.remove(self.filepath)

    def test_basic_chunking(self):
        """Test splitting data into chunks."""

        chunks = list(
            read_csv_chunks(
                self.filepath,
                chunk_size=2,
                delimiter=",",
                skip_header=1
            )
        )

        self.assertEqual(len(chunks), 3)

        self.assertEqual(chunks[0].shape, (2, 2))
        self.assertEqual(chunks[1].shape, (2, 2))
        self.assertEqual(chunks[2].shape, (1, 2))

    def test_exact_chunk_division(self):
        """Test when rows divide exactly by chunk size."""

        chunks = list(
            read_csv_chunks(
                self.filepath,
                chunk_size=5,
                delimiter=",",
                skip_header=1
            )
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].shape, (5, 2))

    def test_uneven_chunks(self):
        """Test when the final chunk is smaller."""

        chunks = list(
            read_csv_chunks(
                self.filepath,
                chunk_size=3,
                delimiter=",",
                skip_header=1
            )
        )

        self.assertEqual(len(chunks), 2)

        self.assertEqual(chunks[0].shape, (3, 2))
        self.assertEqual(chunks[1].shape, (2, 2))

    def test_chunk_size_one(self):
        """Test the smallest valid chunk size."""

        chunks = list(
            read_csv_chunks(
                self.filepath,
                chunk_size=1,
                delimiter=",",
                skip_header=1
            )
        )

        self.assertEqual(len(chunks), 5)

        for chunk in chunks:
            self.assertEqual(chunk.shape, (1, 2))

    def test_chunk_size_larger_than_data(self):
        """Test chunk size larger than number of rows."""

        chunks = list(
            read_csv_chunks(
                self.filepath,
                chunk_size=100,
                delimiter=",",
                skip_header=1
            )
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].shape, (5, 2))

    def test_chunk_size_zero(self):
        """Test invalid chunk size of zero."""

        with self.assertRaises(ValueError):
            list(
                read_csv_chunks(
                    self.filepath,
                    chunk_size=0,
                    delimiter=",",
                    skip_header=1
                )
            )

    def test_negative_chunk_size(self):
        """Test negative chunk size."""

        with self.assertRaises(ValueError):
            list(
                read_csv_chunks(
                    self.filepath,
                    chunk_size=-1,
                    delimiter=",",
                    skip_header=1
                )
            )

    def test_chunks_are_numpy_arrays(self):
        """Test that every chunk is a NumPy array."""

        chunks = list(
            read_csv_chunks(
                self.filepath,
                chunk_size=2,
                delimiter=",",
                skip_header=1
            )
        )

        for chunk in chunks:
            self.assertIsInstance(chunk, np.ndarray)


class TestCSVFiles(unittest.TestCase):
    data_dir = Path(__file__).parent / "data"

    def write_csv(self, content):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "input.csv"
        path.write_text(content, encoding="utf-8")
        return path

    def test_real_csv_contents_and_chunk_equivalence(self):
        path = self.data_dir / "numeric.csv"
        expected = [[170, 65], [180, 80], [175, 70], [165, 60], [172, 68]]
        data = read_csv(path, skip_header=1, dtype=float)
        np.testing.assert_array_equal(data, expected)
        for size in (1, 2, 5, 10):
            with self.subTest(chunk_size=size):
                chunks = list(read_csv_chunks(path, chunk_size=size, skip_header=1, dtype=float))
                np.testing.assert_array_equal(np.vstack(chunks), expected)
                for chunk in chunks:
                    self.assertEqual(chunk.dtype, np.dtype(float))

    def test_missing_values_in_full_file_and_chunks(self):
        path = self.data_dir / "missing.csv"
        for fill in (np.nan, -99):
            with self.subTest(fill=fill):
                options = dict(skip_header=1, dtype=float, filling_values=fill)
                expected = [[170, 65], [180, fill], [fill, 70]]
                np.testing.assert_allclose(read_csv(path, **options), expected, equal_nan=True)
                chunks = list(read_csv_chunks(path, chunk_size=2, **options))
                np.testing.assert_allclose(np.vstack(chunks), expected, equal_nan=True)

    def test_custom_delimiter_missing_marker_and_headers(self):
        path = self.write_csv("metadata\na;b\n1;MISSING\nMISSING;4\n")
        options = dict(delimiter=";", missing_values="MISSING", filling_values=-1,
                       skip_header=2, dtype=float)
        expected = [[1, -1], [-1, 4]]
        np.testing.assert_array_equal(read_csv(path, **options), expected)
        chunks = list(read_csv_chunks(path, chunk_size=1, **options))
        np.testing.assert_array_equal(np.vstack(chunks), expected)

    def test_chunk_shapes_single_column_single_row_and_single_value(self):
        for content, expected in (("1\n2\n3\n", [[1], [2], [3]]),
                                  ("1,2\n", [[1, 2]]), ("7\n", [[7]])):
            with self.subTest(content=content):
                path = self.write_csv(content)
                chunks = list(read_csv_chunks(path, chunk_size=2, dtype=float))
                np.testing.assert_array_equal(np.vstack(chunks), expected)
                self.assertTrue(all(chunk.ndim == 2 for chunk in chunks))

    def test_empty_file_and_header_only(self):
        for content, skip in (("", 0), ("a,b\n", 1), ("a,b\n", 5)):
            with self.subTest(content=content, skip=skip):
                path = self.write_csv(content)
                with self.assertWarns(UserWarning):
                    data = read_csv(path, skip_header=skip)
                self.assertEqual(data.size, 0)
                self.assertEqual(data.ndim, 2)
                self.assertEqual(list(read_csv_chunks(path, chunk_size=2, skip_header=skip)), [])

    def test_malformed_csv(self):
        path = self.data_dir / "malformed.csv"
        with self.assertRaisesRegex(ValueError, "Failed to load CSV") as caught:
            read_csv(path, skip_header=1)
        self.assertIsInstance(caught.exception.__cause__, ValueError)
        with self.assertRaises(ValueError):
            list(read_csv_chunks(path, chunk_size=2, skip_header=1))

    def test_missing_file(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "missing.csv"
        with self.assertRaisesRegex(ValueError, "Failed to load CSV") as caught:
            read_csv(path)
        self.assertIsInstance(caught.exception.__cause__, OSError)
        with self.assertRaises(FileNotFoundError):
            list(read_csv_chunks(path, chunk_size=1))

    def test_omitted_and_noninteger_chunk_size(self):
        path = self.data_dir / "numeric.csv"
        with self.assertRaisesRegex(ValueError, "Chunk size"):
            list(read_csv_chunks(path))
        with self.assertRaises(TypeError):
            list(read_csv_chunks(path, chunk_size=1.5))


if __name__ == "__main__":
    unittest.main()
