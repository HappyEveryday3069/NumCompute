import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from numcompute import benchmarking as bm


class TestBenchmark(unittest.TestCase):
    def test_result_and_keyword_arguments(self):
        result = bm.benchmark(lambda x, offset: x + offset, 3, repeat=2, offset=4)
        self.assertEqual(result['result'], 7)
        self.assertEqual(len(result['times']), 2)
        self.assertTrue({'result', 'times', 'mean', 'min', 'max', 'median'} <= result.keys())

    def test_warmup_excluded_from_times(self):
        calls = []
        ticks = [0, 1000000, 10000000, 13000000]
        with patch.object(bm.time, 'perf_counter_ns', side_effect=ticks):
            result = bm.benchmark(lambda: calls.append(1) or 7, repeat=2)
        self.assertEqual(len(calls), 3)
        np.testing.assert_allclose(result['times'], [0.001, 0.003])
        self.assertAlmostEqual(result['mean'], 0.002)
        self.assertAlmostEqual(result['median'], 0.002)
        self.assertAlmostEqual(result['min'], 0.001)
        self.assertAlmostEqual(result['max'], 0.003)

    def test_invalid_repeat(self):
        for repeat in (0, -1):
            with self.subTest(repeat=repeat), self.assertRaises(ValueError):
                bm.benchmark(lambda: 1, repeat=repeat)
        for repeat in (True, 1.5, '2', None):
            with self.subTest(repeat=repeat), self.assertRaises(TypeError):
                bm.benchmark(lambda: 1, repeat=repeat)

    def test_invalid_callable(self):
        with self.assertRaises(TypeError):
            bm.benchmark(None)

    def test_function_exception_propagates(self):
        def fail():
            raise RuntimeError('objective failed')

        with self.assertRaisesRegex(RuntimeError, 'objective failed'):
            bm.benchmark(fail)


class TestComparisons(unittest.TestCase):
    def test_sum_functions(self):
        x = np.array([2.0, -3.0, 7.0])
        self.assertEqual(bm.python_sum(x), 6)
        self.assertEqual(bm.numpy_sum(x), 6)

    def test_compare_sum_returns_equal_results(self):
        result = bm.compare_sum(n=31, repeat=np.int64(2), seed=7)
        self.assertEqual(set(result), {'python_sum', 'numpy_sum'})
        np.testing.assert_allclose(result['python_sum']['result'], result['numpy_sum']['result'])
        self.assertEqual(len(result['python_sum']['times']), 2)
        self.assertEqual(len(result['numpy_sum']['times']), 2)

    def test_seed_reproduces_data_results(self):
        first = bm.compare_sum(n=31, repeat=1, seed=7)
        second = bm.compare_sum(n=31, repeat=1, seed=7)
        for key in first:
            self.assertEqual(first[key]['result'], second[key]['result'])

    def test_global_rng_not_changed(self):
        before = np.random.get_state()
        bm.compare_sum(n=5, repeat=1, seed=7)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_sum_mismatch_stops_before_timing(self):
        with patch.object(bm, 'numpy_sum', return_value=-1), patch.object(bm, 'benchmark') as timer:
            with self.assertRaises(AssertionError):
                bm.compare_sum(n=5, repeat=1)
            timer.assert_not_called()

    def test_distance_loop_known_results(self):
        np.testing.assert_allclose(bm.python_distances([[3, 4], [0, 0]], [0, 0]), [5, 0])

    def test_compare_distances_returns_equal_results(self):
        result = bm.compare_distances(n=7, repeat=2, seed=7, features=3)
        self.assertEqual(set(result), {'python_distances', 'numpy_distances'})
        np.testing.assert_allclose(result['python_distances']['result'], result['numpy_distances']['result'])
        self.assertEqual(result['numpy_distances']['result'].shape, (7,))
        self.assertEqual(len(result['numpy_distances']['times']), 2)

    def test_distance_mismatch_stops_before_timing(self):
        with patch.object(bm, 'python_distances', return_value=np.zeros(5)), patch.object(bm, 'benchmark') as timer:
            with self.assertRaises(AssertionError):
                bm.compare_distances(n=5, repeat=1)
            timer.assert_not_called()

    def test_invalid_comparison_parameters(self):
        for function in (bm.compare_sum, bm.compare_distances):
            for name in ('n', 'repeat'):
                for value in (0, -1):
                    with self.subTest(function=function.__name__, name=name, value=value), self.assertRaises(ValueError):
                        function(**{name: value})
                for value in (True, 1.5):
                    with self.subTest(function=function.__name__, name=name, value=value), self.assertRaises(TypeError):
                        function(**{name: value})
        for features in (0, -1):
            with self.subTest(features=features), self.assertRaises(ValueError):
                bm.compare_distances(n=2, features=features)


class TestReporting(unittest.TestCase):
    def test_environment_metadata(self):
        info = bm.environment_info()
        self.assertEqual(info['numpy'], np.__version__)
        self.assertEqual(info['time_unit'], 'seconds')
        self.assertTrue(info['python'])
        self.assertTrue(info['platform'])
        self.assertIn('OPENBLAS_NUM_THREADS', info['thread_settings'])
        json.dumps(info)

    def test_table_units_and_speedup(self):
        results = {
            'loop': dict(mean=0.004, median=0.003, min=0.002, max=0.006),
            'numpy': dict(mean=0.002, median=0.002, min=0.001, max=0.003),
        }
        table = bm.format_results(results)
        self.assertIn('Mean (ms)', table)
        self.assertIn('4.0000', table)
        self.assertIn('2.00x', table)

    def test_zero_time_has_no_division_error(self):
        timing = dict(mean=0.0, median=0.0, min=0.0, max=0.0)
        self.assertIn('unavailable', bm.format_results({'loop': timing, 'numpy': timing}))

    def test_command_line_and_json_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'results.json'
            argv = ['benchmarking', '--n', '10', '--distance-n', '4', '--features', '2',
                    '--repeat', '2', '--seed', '7', '--output', str(output)]
            stream = io.StringIO()
            with patch.object(bm.sys, 'argv', argv), contextlib.redirect_stdout(stream):
                bm.main()
            report = json.loads(output.read_text(encoding='utf-8'))
        self.assertIn('Speedup', stream.getvalue())
        self.assertIn('Python:', stream.getvalue())
        self.assertEqual(report['settings']['seed'], 7)
        self.assertEqual(report['settings']['n'], 10)
        self.assertEqual(report['settings']['features'], 2)
        self.assertEqual(len(report['sum']['python_sum']['times']), 2)
        self.assertNotIn('result', report['distances']['numpy_distances'])
        self.assertIn('numpy', report['environment'])


if __name__ == '__main__':
    unittest.main()
