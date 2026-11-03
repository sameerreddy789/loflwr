"""Separate public-data scope detector; never replaces the locked banking head.

Train/calibrate on original-training data only. The confirmation action measures
false accepts on previously unscored OTHER BANKING77 intents, not 20-intent
accuracy or new positive coverage. No CLINC test or old confirmation is read.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time
import warnings

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fedlora_edge.data import INTENTS,SOURCE_SHA256,content_hash,normalize_query,read_source,write_json
from fedlora_edge.encoder_data import load_encoder_manifest,load_feature_split,sha256


def read(path): return json.loads(Path(path).read_text())


def select_negative_data(train,test,train_per_label=25,validation_per_label=10):
    labels=sorted({r["label"] for r in train}-set(INTENTS))
    forbidden={normalize_query(r["query"]) for r in test}
    seen={}
    for r in [*test,*train]:
        key=normalize_query(r["query"])
        if not key or (key in seen and seen[key]!=r["label"]):
            raise ValueError("Empty query or conflicting labels in original source")
        seen[key]=r["label"]
    result={"train":[],"validation":[],"negative_test":[]}
    for label in labels:
        eligible={normalize_query(r["query"]):r for r in train if r["label"]==label and normalize_query(r["query"]) not in forbidden}
        rows=list(eligible.values())
        random.Random(int(content_hash([20261103,label]),16)).shuffle(rows)
        if len(rows)<train_per_label+validation_per_label:
            raise ValueError("Insufficient distinct negative training queries")
        result["validation"].extend(rows[:validation_per_label])
        result["train"].extend(rows[validation_per_label:validation_per_label+train_per_label])
    unique={}
    for r in test:
        if r["label"] in labels: unique.setdefault(normalize_query(r["query"]),r)
    result["negative_test"]=list(unique.values())
    if any(r["source_split"]!="train" for name in ("train","validation") for r in result[name]):
        raise ValueError("Negative development data must originate from training")
    if any(r["source_split"]!="test" for r in result["negative_test"]):
        raise ValueError("Negative test origin mismatch")
    return result


def selection_audit(train,test,selection):
    """Record source accounting before inference, including normalized duplicates."""
    labels={r["label"] for r in selection["train"]}
    seen={}
    duplicates=[]
    for row in test:
        if row["label"] not in labels: continue
        key=normalize_query(row["query"])
        if key in seen:
            duplicates.append({"retained_source_id":seen[key]["source_id"],
                               "excluded_source_id":row["source_id"]})
        else:
            seen[key]=row
    if (len(selection["negative_test"])!=len(seen)
        or {normalize_query(r["query"]) for r in selection["negative_test"]}!=set(seen)):
        raise ValueError("Negative test selection does not match distinct source queries")
    return {"unsupported_intents":len(labels),
            "source_negative_training_rows":sum(r["label"] in labels for r in train),
            "source_negative_test_rows":sum(r["label"] in labels for r in test),
            "selected_counts":{name:len(rows) for name,rows in selection.items()},
            "negative_test_normalized_duplicates":duplicates,
            "deduplication":"Unicode NFKC, case folding, whitespace collapse; first test row retained; all test text excluded from development"}


def coverage_threshold(positive_scores,minimum_coverage=0.95):
    import numpy as np
    import math
    scores=np.asarray(positive_scores)
    if scores.ndim!=1 or len(scores)==0 or not np.isfinite(scores).all() or not 0<minimum_coverage<=1:
        raise ValueError("Invalid positive calibration scores/coverage")
    needed=math.ceil(minimum_coverage*len(scores))
    return float(np.sort(scores)[len(scores)-needed])


def freeze(output):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    from fedlora_edge.encoder_local_data import encode_local_records
    from tools.evaluate_external_clinc import original_files
    if output.exists(): raise FileExistsError("Use a fresh scope study directory")
    original,files=original_files()
    for split in ("train","test"):
        path=ROOT/f"assets/banking77/source/{split}.csv"
        if sha256(path)!=SOURCE_SHA256[f"{split}.csv"]: raise ValueError("BANKING77 source changed")
    source_train=read_source(ROOT/"assets/banking77/source/train.csv","train")
    source_test=read_source(ROOT/"assets/banking77/source/test.csv","test")
    selection=select_negative_data(source_train,source_test)
    audit=selection_audit(source_train,source_test,selection)
    if (audit["selected_counts"]!={"train":1425,"validation":570,"negative_test":2279}
        or audit["unsupported_intents"]!=57 or audit["source_negative_test_rows"]!=2280
        or len(audit["negative_test_normalized_duplicates"])!=1):
        raise ValueError("Unexpected scope source counts")
    output.mkdir(parents=True)
    manifest=load_encoder_manifest(ROOT/"assets/encoder-bank20-v1")
    choices={"kind":"public_scope_detector_development_v1","created_utc":datetime.now(timezone.utc).isoformat(),
        "negative_train_per_label":25,"negative_validation_per_label":10,"selection_seed":20261103,
        "C_values":[0.1,1.0,10.0],"minimum_positive_validation_coverage":0.95,
        "selection":"fewest negative validation false accepts with at least 95% positive validation coverage; then most positive accepts; then lower C",
        "negative_test_status":"unscored","original_confirmation_lock":original["lock_hash"],
        "source_accounting":audit,
        "encoder_manifest_hash":manifest["manifest_hash"],"records_hashes":{name:content_hash(rows) for name,rows in selection.items()},
        "scope":"Centralized public-data binary detector on frozen MPNet, separate from federated head. Negative-only final test."}
    write_json(output/"development-plan.json",choices)
    for name,rows in selection.items(): write_json(output/f"{name}.json",rows)
    started=time.perf_counter()
    negatives,telemetry=encode_local_records(selection["train"]+selection["validation"],ROOT/"assets/reference-mpnet",manifest["encoder"])
    pos=[load_feature_split(ROOT/"assets/encoder-bank20-v1",c)[0] for c in manifest["clients"]]
    positive=np.concatenate(pos)
    validation,_,_=load_feature_split(ROOT/"assets/encoder-bank20-v1")
    X=np.concatenate([positive,negatives[:1425]])
    y=np.concatenate([np.ones(len(positive),dtype=np.int64),np.zeros(1425,dtype=np.int64)])
    candidates=[]
    models=[]
    for C in choices["C_values"]:
        model=LogisticRegression(C=C,class_weight="balanced",solver="liblinear",max_iter=2000,random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("error",ConvergenceWarning)
            model.fit(X,y)
        positive_scores=model.decision_function(validation)
        negative_scores=model.decision_function(negatives[1425:])
        threshold=coverage_threshold(positive_scores)
        point={"C":C,"threshold":threshold,"positive_accepted":int((positive_scores>=threshold).sum()),
               "positive_count":len(validation),"negative_false_accepts":int((negative_scores>=threshold).sum()),"negative_count":570}
        candidates.append(point)
        models.append(model)
    selected=min(range(len(candidates)),key=lambda i:(candidates[i]["negative_false_accepts"],-candidates[i]["positive_accepted"],candidates[i]["C"]))
    model=models[selected]
    model_path=output/"scope-model.json"
    write_json(model_path,{"coef":model.coef_[0].tolist(),"intercept":float(model.intercept_[0]),
        "threshold":candidates[selected]["threshold"],"encoder_identity_hash":manifest["encoder_identity_hash"],
        "selection":candidates[selected],"candidates":candidates,"positive_training_count":len(positive),"negative_training_count":1425})
    import sklearn
    result={"status":"development_completed","scope":choices["scope"],"candidates":candidates,
        "selected":candidates[selected],"wall_seconds":time.perf_counter()-started,
        "negative_encoding":telemetry,"sklearn":sklearn.__version__,"python":sys.version}
    write_json(output/"development-result.json",result)
    lock_confirmation(output)
    print(json.dumps(result),flush=True)


def lock_confirmation(output):
    """Finalize hashes after development, without encoding or scoring test queries."""
    from tools.evaluate_external_clinc import original_files
    if any((output/name).exists() for name in ("negative-test-started.json","negative-test-result.json")):
        raise RuntimeError("Cannot create another lock after negative test inference starts")
    original,files=original_files()
    choices=read(output/"development-plan.json")
    development=read(output/"development-result.json")
    model_path=output/"scope-model.json"
    model=read(model_path)
    dataset=ROOT/"assets/encoder-bank20-v1"
    manifest=load_encoder_manifest(dataset)
    if (development["status"]!="development_completed" or development["selected"]!=model["selection"]
        or choices["encoder_manifest_hash"]!=manifest["manifest_hash"]
        or model["encoder_identity_hash"]!=manifest["encoder_identity_hash"]
        or choices["negative_test_status"]!="unscored"):
        raise ValueError("Scope development artifacts do not agree")
    for name,digest in choices["records_hashes"].items():
        if content_hash(read(output/f"{name}.json"))!=digest:
            raise ValueError("Scope selected record manifest changed")
    paths=[model_path,output/"negative_test.json",output/"train.json",output/"validation.json",
           output/"development-plan.json",output/"development-result.json",Path(__file__),
           ROOT/"tests/test_scope_study.py",ROOT/"fedlora_edge/encoder_local_data.py",
           ROOT/"tools/evaluate_external_clinc.py",dataset/"manifest.json"]
    for entry in [*manifest["clients"].values(),manifest["validation"]]:
        path=dataset/entry["file"]
        if sha256(path)!=entry["sha256"]:
            raise ValueError("Positive development feature archive changed")
        paths.append(path)
    for path in paths:
        files[path.relative_to(ROOT).as_posix()]=sha256(path)
    plan={"kind":"locked_negative_only_scope_confirmation_v1","run_id":output.name,
        "files":files,"model_file":model_path.relative_to(ROOT).as_posix(),
        "model_sha256":sha256(model_path),
        "negative_test_file":(output/"negative_test.json").relative_to(ROOT).as_posix(),
        "negative_test_hash":choices["records_hashes"]["negative_test"],
        "negative_test_count":choices["source_accounting"]["selected_counts"]["negative_test"],
        "source_accounting":choices["source_accounting"],
        "head_hash":original["head_hash"],"margin_thresholds":[p["threshold"] for p in original["selective_policies"]["policies"]],
        "model_selection":development["selected"],"scope":choices["scope"],"status":"locked_before_negative_test_inference"}
    plan["lock_hash"]=content_hash(plan)
    write_json(output/"confirmation-plan.json",plan)
    return plan


def evaluate(output):
    import numpy as np
    from fedlora_edge.encoder_local_data import encode_local_records
    from fedlora_edge.encoder_head import HeadTrainer
    from fedlora_edge.encoder_inference import load_head_checkpoint
    import subprocess
    plan=read(output/"confirmation-plan.json")
    def verify():
        if content_hash({k:v for k,v in plan.items() if k!="lock_hash"})!=plan["lock_hash"]:
            raise ValueError("Scope lock changed")
        for name,digest in plan["files"].items():
            if sha256(ROOT/name)!=digest: raise ValueError(f"Scope frozen file changed: {name}")
    verify()
    if (output/"negative-test-result.json").exists(): return read(output/"negative-test-result.json")
    public=ROOT/"research/results/scope-v1/confirmation-plan.json"
    if read(public)!=plan: raise ValueError("Publish the exact scope confirmation plan before evaluation")
    git=["git","-c",f"safe.directory={ROOT.as_posix()}"]
    committed=json.loads(subprocess.check_output([*git,"show","HEAD:research/results/scope-v1/confirmation-plan.json"],cwd=ROOT))
    if committed!=plan: raise ValueError("Commit the exact scope lock before inference")
    commit=subprocess.check_output([*git,"rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    if (output/"negative-test-started.json").exists(): raise RuntimeError("Incomplete scope evaluation requires investigation")
    records=read(ROOT/plan["negative_test_file"])
    if (content_hash(records)!=plan["negative_test_hash"] or len(records)!=plan["negative_test_count"]
        or len({normalize_query(r["query"]) for r in records})!=len(records)
        or any(r["label"] in INTENTS for r in records)):
        raise ValueError("Negative test differs from the locked unsupported ontology")
    model=read(ROOT/plan["model_file"])
    manifest=load_encoder_manifest(ROOT/"assets/encoder-bank20-v1")
    write_json(output/"negative-test-started.json",{"lock_hash":plan["lock_hash"],"commit":commit,"time":time.time()})
    X,telemetry=encode_local_records(records,ROOT/"assets/reference-mpnet",manifest["encoder"])
    scope_scores=X@np.array(model["coef"])+model["intercept"]
    passes=scope_scores>=model["threshold"]
    head=HeadTrainer()
    head.set_state(load_head_checkpoint(ROOT/"runs/encoder-flower-balanced-v2/selected")["state"])
    logits=head.predict_scores(X)
    ordered=np.sort(logits,axis=1)
    margins=ordered[:,-1]-ordered[:,-2]
    comparisons=[]
    for threshold in [0.0,*plan["margin_thresholds"]]:
        accepted=margins>=threshold
        combined=accepted&passes
        comparisons.append({"margin_threshold":threshold,"count":len(records),
            "original_margin_false_accepts":int(accepted.sum()),"with_scope_false_accepts":int(combined.sum()),
            "original_margin_false_accept_rate":float(accepted.mean()),"with_scope_false_accept_rate":float(combined.mean())})
    result={"status":"completed","run_id":plan["run_id"],"lock_hash":plan["lock_hash"],"pre_inference_commit":commit,
        "count":len(records),"unsupported_intents":57,"scope_only_false_accepts":int(passes.sum()),
        "comparisons":comparisons,"encoding":telemetry,
        "interpretation":"Negative-only held-out test. Positive coverage is validation-only. No new 20-intent accuracy or full-system confirmation.",
        "per_intent":{label:{"count":sum(r["label"]==label for r in records),
            "scope_false_accepts":sum(bool(p) and r["label"]==label for r,p in zip(records,passes))} for label in sorted({r["label"] for r in records})}}
    verify()
    np.savez(output/"negative-test-scores.npz",scope=scope_scores,head=logits)
    write_json(output/"negative-test-result.json",result)
    print(json.dumps({k:result[k] for k in ("status","count","comparisons","interpretation")}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["develop","lock","evaluate"])
    parser.add_argument("--output",type=Path,default=ROOT/"runs/scope-v1")
    args=parser.parse_args()
    {"develop":freeze,"lock":lock_confirmation,"evaluate":evaluate}[args.action](args.output.resolve())


if __name__=="__main__": main()
