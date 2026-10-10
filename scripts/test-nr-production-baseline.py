#!/usr/bin/env python3
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('baseline',Path(__file__).with_name('audit-nr-production-baseline.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
PEER=Path(__file__).resolve().parents[1].parent/'NeuralStudio-nr-fable'
RUN=PEER/'build/nr_full/baseline-prod-v1'
TARGET=RUN/'v11-p1/process-1/receipt.json'


class ProductionBaseline(unittest.TestCase):
    def reject(self,edit,target=TARGET):
        original=m.read
        def altered(path):
            r=original(path)
            if Path(path)==target:r=copy.deepcopy(r);edit(r)
            return r
        with patch.object(m,'read',side_effect=altered):
            with self.assertRaises(ValueError):m.audit(PEER,RUN)

    def test_incomplete_process(self):self.reject(lambda r:r['frames'].pop())
    def test_wrong_poison(self):self.reject(lambda r:r['frames'][0].update(poison=90))
    def test_missing_write(self):self.reject(lambda r:r['frames'][0]['body'].update(final_pixels_not_written=1))
    def test_stale_history(self):self.reject(lambda r:r['frames'][1]['body'].update(history_sha256_before='0'*64))
    def test_hidden_omission(self):self.reject(lambda r:r['frames'][0]['body'].update(checks_skipped=[]))
    def test_missing_provenance(self):self.reject(lambda r:r['frames'][0]['body']['loaded'].pop(next(iter(r['frames'][0]['body']['loaded']))))
    def test_stale_telemetry(self):self.reject(lambda r:r['frames'][0]['encoder_telemetry'].update(generation=2))
    def test_old_clock_field(self):self.reject(lambda r:r['frames'][0].update(wall_ms_total=100000))


if __name__=='__main__':unittest.main()
