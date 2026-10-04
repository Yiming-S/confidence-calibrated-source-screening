"""Current-cohort aggregate and certificate regression checks."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from verify_kumar_results import DEFAULT_INPUT, verify_kumar_results


class KumarResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)/'kumar2024'
        shutil.copytree(DEFAULT_INPUT,self.folder)

    def test_reconstructs_without_changing_inputs(self):
        before={p.relative_to(self.folder):p.read_bytes() for p in self.folder.rglob('*') if p.is_file()}
        report=verify_kumar_results(self.folder)
        self.assertEqual(report['subject_method_rows'],108)
        self.assertEqual(report['paired_intervals'],2)
        self.assertEqual(before,{p.relative_to(self.folder):p.read_bytes() for p in self.folder.rglob('*') if p.is_file()})

    def test_changed_interval_is_rejected(self):
        p=self.folder/'summary.json'
        data=json.loads(p.read_text())
        data['primary_refinement_minus_all_sources']['two_sided_95_ci'][0]+=.01
        p.write_text(json.dumps(data))
        with self.assertRaises(ValueError):verify_kumar_results(self.folder)

    def test_changed_component_endpoint_is_rejected(self):
        p=self.folder/'supplemental/certificate.json'
        data=json.loads(p.read_text());data['sources'][0]['lower_component']+=.1
        p.write_text(json.dumps(data))
        with self.assertRaises(ValueError):verify_kumar_results(self.folder)

    def test_changed_exploratory_role_is_rejected(self):
        p=self.folder/'summary.json'
        data=json.loads(p.read_text());data['analysis_role']='confirmatory'
        p.write_text(json.dumps(data))
        with self.assertRaises(ValueError):verify_kumar_results(self.folder)

if __name__=='__main__':unittest.main()
