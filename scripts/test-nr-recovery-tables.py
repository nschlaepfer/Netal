#!/usr/bin/env python3
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('tables', Path(__file__).with_name('verify-nr-recovery-tables.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
DIRECTORY = m.ROOT / 'build/nr_rcp_half/rtx-v1'
SOURCE = m.ROOT / 'build/nr_rcp_half/source-v1/rsq_half_oracle.cu'
EXPECTED = 'dfb76ea177f91a47247b7096829e2b40ce05cf0833749066574a97c56eb568f4'


class RecoveryTables(unittest.TestCase):
    def run_table(self):
        return m.half_table(DIRECTORY, SOURCE, 'rcp', EXPECTED)

    def reject(self, change, filename='output/5a-b.json'):
        original = m.read
        def mutated(path):
            result = original(path)
            if path == DIRECTORY / filename:
                result = copy.deepcopy(result)
                change(result)
            return result
        with patch.object(m, 'read', side_effect=mutated):
            with self.assertRaises(ValueError):
                self.run_table()

    def test_retained_half_domain(self):
        self.assertEqual(self.run_table()['bytes'], 63486)

    def test_wrong_instruction(self):
        self.reject(lambda r: r.update(instruction='rcp.approx.ftz.f32'))

    def test_wrong_poison(self):
        self.reject(lambda r: r.update(poison=165))

    def test_duplicate_process(self):
        pid = m.read(DIRECTORY / 'output/a5-a.json')['pid']
        self.reject(lambda r: r.update(pid=pid))

    def test_expected_tensor_input(self):
        self.reject(lambda r: r.update(expected_input_opened=True))

    def test_guard_error(self):
        self.reject(lambda r: r.update(errors=1))

    def test_incomplete_domain(self):
        self.reject(lambda r: r.update(count=31742))

    def test_wrong_source(self):
        self.reject(lambda r: r.update(source_sha256='0' * 64), 'receipt.json')

    def test_duplicate_artifact(self):
        self.reject(lambda r: r['artifacts'].__setitem__(0, r['artifacts'][1]), 'receipt.json')

    def test_changed_byte_pin(self):
        self.reject(lambda r: r['artifacts'][0].update(sha256='0' * 64), 'receipt.json')


if __name__ == '__main__':
    unittest.main()
