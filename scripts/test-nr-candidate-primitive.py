#!/usr/bin/env python3
"""Negative controls for artifact-to-primitive binding; no GPU work."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import nr_candidate_primitive as gate

class PrimitiveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='nr-primitive-gate-')
        self.addCleanup(self.temp.cleanup);self.bundle=Path(self.temp.name)
        self.proof=json.loads((Path(__file__).resolve().parents[1]/'evidence/specs/nr-encoder-fscale-v1/primitive-verification.json').read_text())
    def write_proof(self):(self.bundle/'primitive-verification.json').write_text(json.dumps(self.proof))
    def test_legacy_accepts_only_pinned_header(self):
        with patch.object(gate,'sha',return_value=gate.HEADERS[gate.LEGACY_ROOT]):
            self.assertEqual(gate.validate_primitive(self.bundle,{'primitive_root':gate.LEGACY_ROOT}),gate.LEGACY_ROOT)
    def test_changed_legacy_header_rejected(self):
        with patch.object(gate,'sha',return_value='0'*64),self.assertRaises(ValueError):gate.validate_primitive(self.bundle,{'primitive_root':gate.LEGACY_ROOT})
    def test_unknown_root_rejected(self):
        with self.assertRaises(ValueError):gate.validate_primitive(self.bundle,{'primitive_root':'0'*64})
    def test_new_root_and_exact_proof(self):
        self.write_proof()
        with patch.object(gate,'sha',return_value=gate.HEADERS[gate.FSCALE_ROOT]):
            self.assertEqual(gate.validate_primitive(self.bundle,{'primitive_root':gate.FSCALE_ROOT}),gate.FSCALE_ROOT)
    def test_missing_new_proof(self):
        with patch.object(gate,'sha',return_value=gate.HEADERS[gate.FSCALE_ROOT]),self.assertRaises(FileNotFoundError):gate.validate_primitive(self.bundle,{'primitive_root':gate.FSCALE_ROOT})
    def test_tampered_proof(self):
        self.proof['result']='fail';self.write_proof()
        with patch.object(gate,'sha',return_value=gate.HEADERS[gate.FSCALE_ROOT]),self.assertRaises(ValueError):gate.validate_primitive(self.bundle,{'primitive_root':gate.FSCALE_ROOT})
    def test_mixed_headers_rejected(self):
        self.write_proof()
        with patch.object(gate,'sha',side_effect=[gate.HEADERS[gate.FSCALE_ROOT],gate.HEADERS[gate.LEGACY_ROOT]]),self.assertRaises(ValueError):gate.validate_primitive(self.bundle,{'primitive_root':gate.FSCALE_ROOT})

if __name__=='__main__':unittest.main()
