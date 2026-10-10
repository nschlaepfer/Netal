#!/usr/bin/env python3
"""CPU checks for the fail-closed source extraction helpers (no GPU)."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('builder',Path(__file__).with_name('build-nr-encoder-runtime.py'))
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)

class Extraction(unittest.TestCase):
    def test_unique_replacement(self):self.assertEqual(builder.once('before TARGET after','TARGET','NEW'),'before NEW after')
    def test_missing_anchor(self):
        with self.assertRaises(ValueError):builder.once('before after','TARGET','NEW')
    def test_duplicated_anchor(self):
        with self.assertRaises(ValueError):builder.once('TARGET TARGET','TARGET','NEW')
    def test_exact_range_preserves_suffix(self):self.assertEqual(builder.between('A START old END Z','START','END','new '),'A new END Z')
    def test_missing_range(self):
        with self.assertRaises(ValueError):builder.between('START old','START','END','new')
    def test_reversed_range(self):
        with self.assertRaises(ValueError):builder.between('END START','START','END','new')
    def test_ambiguous_range(self):
        with self.assertRaises(ValueError):builder.between('START START END','START','END','new')
    def test_diagnostics_do_not_change_dispatch(self):
        line='[encoder setBuffer:y.metal offset:Guard atIndex:2]; [encoder dispatchThreads:size threadsPerThreadgroup:group];'
        self.assertEqual(builder.diagnostic(line),line)
    def test_diagnostic_output_is_gated(self):
        out=builder.diagnostic('fs::create_directories(output);std::ofstream receipt(output/"run.json");')
        self.assertEqual(out,'if(NrEmitDiagnostics)fs::create_directories(output);NrDiagnosticStream receipt(output/"run.json");')
    def test_quiet_output_preserves_dispatch(self):
        line='std::cout<<"stage";[encoder dispatchThreads:size threadsPerThreadgroup:group];'
        self.assertEqual(builder.quiet_progress(line,1),'if(NrEmitDiagnostics)'+line)
    def test_quiet_output_inventory_fails_closed(self):
        with self.assertRaises(ValueError):builder.quiet_progress('std::cout<<"a";std::cout<<"b";',1)
    def test_scratch_rewrite_keeps_all_dispatch_and_validation(self):
        source=(Path(__file__).resolve().parents[1]/'build/nr_encoder_runtime/build-v1/source/generated/pre.inc').read_text()
        candidate=builder.reuse_block0_scratch(source)
        for marker in ('dispatchThreads:','Complete(cmd);','GuardOutput(', 'Canary(', 'Sha('):
            self.assertGreaterEqual(candidate.count(marker),source.count(marker))
        self.assertEqual(candidate.count('ScratchMake(device,'),6)
        self.assertIn('std::memset(Data(b),fill,bytes)',candidate)
        self.assertIn('scratch_allocations!=44||scratch_reuses!=1452',candidate)
        # Returned allocations and full-image counts must not enter the pool.
        self.assertIn('Buffer skip=Make(device,70778880,poison)',candidate)
        self.assertIn('return {skip,encoder,inputHash,featureHash}',candidate)
    def test_scratch_rewrite_rejects_missing_or_repeated_application(self):
        with self.assertRaises(ValueError):builder.reuse_block0_scratch('unknown source')
        source=(Path(__file__).resolve().parents[1]/'build/nr_encoder_runtime/build-v1/source/generated/pre.inc').read_text()
        with self.assertRaises(ValueError):builder.reuse_block0_scratch(builder.reuse_block0_scratch(source))

if __name__=='__main__':unittest.main(verbosity=2)
