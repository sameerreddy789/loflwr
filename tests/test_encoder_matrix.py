import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fedlora_edge.data import INTENTS, content_hash, write_json
from tools.run_encoder_matrix import cells, HEAD, run_cell


class MatrixPlanTests(unittest.TestCase):
    def test_33_unique_matched_cells(self):
        rows=cells()
        self.assertEqual(len(rows),33)
        self.assertEqual(len({r["id"] for r in rows}),33)
        for seed in (42,43,44):
            subset=[r for r in rows if r["seed"]==seed]
            self.assertEqual(len(subset),11)
            base=next(r for r in subset if r["id"]==f"s{seed}-expanded-heterogeneous-equal")
            omission=next(r for r in subset if r["omitted"])
            self.assertEqual(base["budgets"],omission["budgets"])
            self.assertEqual(omission["omitted"],["4"])
            self.assertEqual(sum(e["end"]-e["start"]+1 for e in subset[-1]["unavailable"]),30)

    def test_real_optimizer_parity_with_existing_protocol_runner(self):
        try:
            import numpy as np
            import torch
        except ImportError:
            self.skipTest("Torch and NumPy required for optimizer parity fixture")
        from fedlora_edge.encoder_head import HeadTrainer,HeadConfig
        from fedlora_edge.encoder_data import sha256
        from fedlora_edge.state import hash_state,state_to_dict
        from tools.run_encoder_study import run
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            dataset=root/"data"
            dataset.mkdir()
            entries={}
            for i in range(6):
                X=np.zeros((4,768),dtype=np.float32)
                for j in range(4): X[j,(i%5)*4+j]=1
                y=np.arange((i%5)*4,(i%5)*4+4,dtype=np.int64)
                path=dataset/f"{i}.npz"
                np.savez(path,X=X,y=y,source_ids=np.asarray([f"fixture-{i}-{j}" for j in range(4)]))
                entries[str(i)]={"count":4,"file":path.name,"sha256":sha256(path),"records_hash":str(i)*64}
            manifest={"format_version":1,"kind":"mpnet_local_feature_dataset","feature_dim":768,"labels":list(INTENTS),
                "encoder_identity_hash":"a"*64,"model_revision":"b"*40,
                "clients":{str(i):entries[str(i)] for i in range(5)},"validation":entries["5"]}
            manifest["manifest_hash"]=content_hash(manifest)
            write_json(dataset/"manifest.json",manifest)
            matrix=root/"matrix"
            matrix.mkdir()
            state=HeadTrainer(seed=42,config=HeadConfig(**HEAD)).get_state()
            write_json(matrix/"initial.json",{"state":state_to_dict(state),"state_hash":hash_state(state)})
            plan={"head_config":HEAD,"initial":{"42":{"file":"initial.json","state_hash":hash_state(state)}},
                "datasets":{"expanded":str(dataset)},"attempts":2,"evaluate_every":1,"quorum":3,"scope":"fixture"}
            plan["plan_hash"]=content_hash(plan)
            write_json(matrix/"plan.json",plan)
            cell=next(r for r in cells() if r["id"]=="s42-expanded-heterogeneous-equal")
            with patch("tools.run_encoder_matrix.verify"):
                result=run_cell(str(matrix),cell)
            reference=run({"dataset_dir":str(dataset),"run_dir":str(root/"reference"),"head_config":HEAD,
                "seed":42,"attempts":2,"evaluate_every":1,"quorum":3,"strategy":"equal","budgets":cell["budgets"]})
            self.assertEqual(result["initial_state_hash"],reference["plan"]["initial_state_hash"])
            self.assertEqual(result["final"]["state_hash"],reference["final"]["state_hash"])
            self.assertEqual(result["final"]["metrics"],reference["final"]["metrics"])
            self.assertEqual(result["completed_steps"],{c:n*2 for c,n in cell["budgets"].items()})
            with patch("tools.run_encoder_matrix.verify"):
                self.assertEqual(run_cell(str(matrix),cell),result)
                bad=copy.deepcopy(cell)
                bad["omitted"]=["4"]
                with self.assertRaises(ValueError): run_cell(str(matrix),bad)


if __name__ == "__main__":
    unittest.main()
