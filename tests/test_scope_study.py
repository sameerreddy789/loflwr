import unittest
from pathlib import Path
import tempfile
from tools.run_scope_study import select_negative_data,selection_audit,coverage_threshold,lock_confirmation


class ScopeDataTests(unittest.TestCase):
    def test_training_selection_excludes_test_text_and_target_labels(self):
        train=[{"query":f"fixture other {i}","label":"other_intent","source_id":f"train:{i+1}","source_split":"train"} for i in range(6)]
        train.append({"query":"fixture target","label":"card_arrival","source_id":"train:100","source_split":"train"})
        test=[{"query":"FIXTURE OTHER 0","label":"other_intent","source_id":"test:1","source_split":"test"}]
        splits=select_negative_data(train,test,2,1)
        self.assertEqual({k:len(v) for k,v in splits.items()},{"train":2,"validation":1,"negative_test":1})
        self.assertTrue(all(r["query"]!="fixture other 0" for k in ("train","validation") for r in splits[k]))
        self.assertTrue(all(r["label"]=="other_intent" for rows in splits.values() for r in rows))

    def test_conflicting_labels_fail(self):
        with self.assertRaises(ValueError):
            select_negative_data([{"query":"same","label":"one"}],[{"query":"SAME","label":"two"}],1,1)

    def test_test_duplicates_are_counted_and_first_row_retained(self):
        train=[{"query":f"fixture development {i}","label":"other_intent",
                "source_id":f"train:{i+1}","source_split":"train"} for i in range(35)]
        test=[{"query":q,"label":"other_intent","source_id":f"test:{i+1}","source_split":"test"}
              for i,q in enumerate(["fixture held out", "  FIXTURE held   out ", "second held out"])]
        splits=select_negative_data(train,test)
        audit=selection_audit(train,test,splits)
        self.assertEqual(audit["selected_counts"],{"train":25,"validation":10,"negative_test":2})
        self.assertEqual(audit["source_negative_test_rows"],3)
        self.assertEqual(audit["negative_test_normalized_duplicates"],
                         [{"retained_source_id":"test:1","excluded_source_id":"test:2"}])
        self.assertEqual([r["source_id"] for r in splits["negative_test"]],["test:1","test:3"])
        self.assertEqual(len({r["source_id"] for k in ("train","validation") for r in splits[k]}),35)
        splits["negative_test"].pop()
        with self.assertRaises(ValueError): selection_audit(train,test,splits)

    def test_scope_lock_cannot_be_recreated_after_inference_starts(self):
        for marker in ("negative-test-started.json","negative-test-result.json"):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as directory:
                output=Path(directory)
                (output/marker).write_text("{}")
                with self.assertRaisesRegex(RuntimeError,"after negative test inference"):
                    lock_confirmation(output)

    def test_coverage_threshold_preserves_at_least_required_positives(self):
        try: import numpy as np
        except ImportError: self.skipTest("NumPy optional dependency")
        for values in (np.arange(200),np.zeros(200)):
            threshold=coverage_threshold(values)
            self.assertGreaterEqual(int((values>=threshold).sum()),190)
        for values in ([],[float("nan")]):
            with self.assertRaises(ValueError): coverage_threshold(values)


if __name__=="__main__": unittest.main()
