"""Prepare and audit a blinded human collection; never import or run a model.

All data products belong in ignored assets/ or runs/. The final gate means ready
for a future evaluation lock, not scored, independently verified, or deployable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import date


ROOT = Path(__file__).resolve().parents[1]
VERSION = "fresh-bank20-collection-v1"
NORMALIZATION = "unicode-nfkc-casefold-whitespace-v1"
TARGET_PER_INTENT = 20
# Authored definitions, not examples or model answers. Labels match evaluation.md.
ONTOLOGY = [
    ("card_arrival", "Card lifecycle", "An ordered card has not arrived or its current delivery status is unknown."),
    ("card_delivery_estimate", "Card lifecycle", "A request for the expected delivery time or delivery options for a card."),
    ("activate_my_card", "Card lifecycle", "How to activate a newly received card."),
    ("card_about_to_expire", "Card lifecycle", "A card is approaching expiry and needs renewal or replacement."),
    ("pending_card_payment", "Card payments", "A card purchase remains pending rather than completed."),
    ("declined_card_payment", "Card payments", "An attempted card purchase was refused or failed."),
    ("card_payment_not_recognised", "Card payments", "The customer does not recognize a card purchase."),
    ("transaction_charged_twice", "Card payments", "A single purchase appears to have been charged more than once."),
    ("pending_transfer", "Transfers", "A money transfer is shown as pending."),
    ("transfer_not_received_by_recipient", "Transfers", "A recipient has not received a transfer that was sent."),
    ("transfer_fee_charged", "Transfers", "A question or complaint about a fee charged for a transfer."),
    ("cancel_transfer", "Transfers", "A request to cancel or reverse a transfer."),
    ("pending_cash_withdrawal", "Cash withdrawals", "An ATM cash withdrawal is shown as pending."),
    ("declined_cash_withdrawal", "Cash withdrawals", "An attempted ATM cash withdrawal was refused or failed."),
    ("wrong_amount_of_cash_received", "Cash withdrawals", "An ATM dispensed an amount different from the amount requested or charged."),
    ("cash_withdrawal_not_recognised", "Cash withdrawals", "The customer does not recognize an ATM cash withdrawal."),
    ("edit_personal_details", "Account verification", "A request to change the customer's personal profile details."),
    ("verify_my_identity", "Account verification", "How to complete identity verification, without reporting a verification failure."),
    ("unable_to_verify_identity", "Account verification", "An attempt to verify identity failed or cannot be completed."),
    ("why_verify_identity", "Account verification", "A question about why identity verification is required."),
]
INTENTS = tuple(row[0] for row in ONTOLOGY)
LABELS = frozenset((*INTENTS, "ambiguous", "unsupported"))
OVERLAP_CATEGORIES = (
    "training", "development", "bank20-test-500", "bank20-confirmation-300",
    "external-clinc-269", "scope-negative-2279",
)
QUERY_FIELDS = (
    "query_id", "query", "author_id", "source_id", "collected_at", "collection_batch",
    "provenance", "provenance_detail",
)
PARTICIPANT_FIELDS = (
    "participant_id", "roles", "no_model_outputs", "no_old_queries", "independent_reviews",
)
LABEL_FIELDS = ("query_id", "query", "reviewer_id", "label", "notes")
ADJUDICATION_FIELDS = (
    "query_id", "adjudicator_id", "final_label", "decision", "exclusion_reason",
    "near_duplicate_of", "privacy_checked", "near_duplicate_checked", "reason",
)
PROVENANCES = {"independent_human", "external", "model_generated", "unknown"}
ROLES = {"writer", "collector", "labeler", "adjudicator"}


def normalize_query(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, fields: tuple, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def private_output(path: Path) -> Path:
    """CLI outputs must resolve inside the repository's ignored data directories."""
    resolved = path.resolve()
    if not any(resolved.is_relative_to((ROOT / name).resolve()) for name in ("runs", "assets")):
        raise ValueError("Output must be inside this repository's ignored runs/ or assets/ directory")
    return resolved


def read_rows(path: Path, fields: tuple | None = None, *, records_key: str = "records") -> list[dict]:
    """Strict imports for new records; old sources may retain their existing fields."""
    content = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".csv":
        reader = csv.DictReader(io.StringIO(content))
        header = reader.fieldnames or []
        if len(header) != len(set(header)) or (fields is not None and set(header) != set(fields)):
            raise ValueError(f"{path.name}: duplicate or unexpected CSV columns")
        rows = list(reader)
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        rows = json.loads(content)
        if isinstance(rows, dict):
            for key in records_key.split("."):
                if not key or not isinstance(rows, dict) or key not in rows:
                    raise ValueError(f"{path.name}: records_key does not identify a list")
                rows = rows[key]
    else:
        raise ValueError(f"{path.name}: expected CSV, JSON, or JSONL")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path.name}: records must be objects in a list")
    for index, row in enumerate(rows, 1):
        if fields is not None and set(row) != set(fields):
            raise ValueError(f"{path.name}: row {index} has missing or unexpected fields")
        if fields is not None and any(not isinstance(value, str) for value in row.values()):
            raise ValueError(f"{path.name}: row {index} values must be strings")
        if None in row:
            raise ValueError(f"{path.name}: row {index} has extra CSV values")
    return rows


def ontology_document() -> dict:
    return {"version": VERSION, "intents": [dict(zip(("label", "domain", "definition"), row))
                                             for row in ONTOLOGY],
            "other_labels": {"ambiguous": "Insufficient detail or more than one plausible intent; do not guess.",
                             "unsupported": "A request outside the twenty defined intents."}}


def create_pack(output: Path, batch_id: str) -> None:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,63}", batch_id):
        raise ValueError("batch_id must be a lowercase pseudonymous slug, 3 to 64 characters")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "ontology.json", ontology_document())
    write_json(output / "collection-plan.json", {
        "version": VERSION, "batch_id": batch_id, "target_per_intent": TARGET_PER_INTENT,
        "target_population": "Independently written English banking-support requests",
        "normalization": NORMALIZATION, "protocol_reviewed": False,
        "overlap_inventory_reviewed": False, "curator_id": "",
        "status": "human_collection_pending",
    })
    write_json(output / "overlap-sources.json", [
        {"dataset_id": category, "category": category, "path": "", "text_field": "query", "records_key": "records"}
        for category in OVERLAP_CATEGORIES
    ])
    for name, fields in (("queries", QUERY_FIELDS), ("participants", PARTICIPANT_FIELDS),
                         ("labels-a", LABEL_FIELDS), ("labels-b", LABEL_FIELDS),
                         ("adjudications", ADJUDICATION_FIELDS)):
        write_csv(output / f"{name}.csv", fields, [])
    write_json(output / "evaluation-handoff.json", {
        "status": "pending_future_pre_inference_lock", "evaluation_ready": False,
        "candidate_and_policy_lock": None, "data_lock": None,
        "required": [
            "Commit protocol, candidate model and tokenizer hashes, preprocessing and ordered label mapping.",
            "Freeze the exact combined margin AND scope rule, numeric thresholds, boundary behavior and rejection action.",
            "Hash classifier and scope artifacts and source files; preserve all existing locked evaluations.",
            "Freeze this collection and every retained/excluded record with reasons and supplied overlap-source hashes.",
            "Implement a separate audited evaluator with fixture tests and a once-only start marker before inference.",
            "Commit and push the public lock before running that future evaluator once; keep raw queries private.",
            "Report supported accuracy, macro F1, per-intent/domain results, answered accuracy, coverage and wrong/declined counts.",
            "Report unsupported and ambiguous rejection separately, all-row accounting, uncertainty and author diversity.",
        ],
        "limitations": "Collection readiness is not an evaluation lock, an accuracy result, or deployment approval.",
    })
    instructions = """# Blinded fresh human collection (no queries collected yet)

Read docs/fresh-query-collection.md before recruitment. Freeze and attest the
collection protocol before writing. Target 20 valid fresh supported queries per
intent (400 total); retain every submitted ambiguous, unsupported, external or
excluded row. This tool never scores a model or contacts contributors.

Writers: use only ontology.json and natural English requests. Do not use a model,
old benchmark questions/errors, paraphrases of those questions, names, contact
details, account/card numbers, identity documents, passwords or other credentials.
Use stable p- pseudonyms for people, q- IDs for queries, and src- source IDs.
Never store the identity-to-pseudonym mapping here.

Curator: record provenance/date/source and participant attestations. Reviewers get
only their own generated packet; do not share author identity, intended intent,
provenance, the other reviewer's labels, model outputs or old benchmark queries.
Collect both independent reviews before releasing them to a separate adjudicator.
The adjudicator resolves every row, including disagreements and exclusions,
checks privacy and manually checks near duplicates before any future inference.

Use prepare-labels only after queries are complete. Copy the completed packet
files to labels-a.csv and labels-b.csv without editing query text. Reissuing a
packet requires a new output directory and a documented restart of review.
Use ready-check to list unfinished work; blocked means stop before evaluation.
No score or metric result is produced. Human collection is still pending.
"""
    (output / "INSTRUCTIONS.md").write_text(instructions, encoding="utf-8", newline="\n")


def prepare_labels(queries_path: Path, output: Path) -> None:
    queries = read_rows(queries_path, QUERY_FIELDS)
    seen = set()
    for row in queries:
        if not re.fullmatch(r"q-[a-z0-9_-]{1,60}", row["query_id"]) or not row["query"].strip():
            raise ValueError("Queries need a valid q- pseudonym and nonempty text")
        if row["query_id"] in seen:
            raise ValueError("Duplicate query_id")
        seen.add(row["query_id"])
    if not queries:
        raise ValueError("No human queries exist; cannot prepare label packets")
    output.mkdir(parents=True, exist_ok=False)
    rows = [{"query_id": row["query_id"], "query": row["query"], "reviewer_id": "", "label": "", "notes": ""}
            for row in queries]
    for name in ("reviewer-a", "reviewer-b"):
        directory = output / name
        directory.mkdir()
        write_csv(directory / "labels.csv", LABEL_FIELDS, rows)
        write_json(directory / "ontology.json", ontology_document())
        (directory / "INSTRUCTIONS.md").write_text(
            "Label independently using ontology.json. Use ambiguous for insufficient/multiple meanings; "
            "unsupported for other topics. Do not consult a model, old benchmark questions, author, or "
            "other reviewer. Enter your p- pseudonym, label and optional notes. Keep query text and IDs unchanged.\n",
            encoding="utf-8", newline="\n")
    write_json(output / "packet-manifest.json", {
        "version": VERSION, "query_file_sha256": file_hash(queries_path),
        "query_ids_and_text_sha256": digest([(r["query_id"], r["query"]) for r in queries]),
        "count": len(rows), "contains_model_outputs": False,
    })


def overlap_index(sources: list[dict], base: Path) -> tuple[dict[str, list[str]], list[dict], list[str]]:
    hashes: dict[str, list[str]] = defaultdict(list)
    evidence, errors, categories, dataset_ids = [], [], set(), set()
    if not isinstance(sources, list):
        return {}, [], ["overlap_sources_not_a_list"]
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"dataset_id", "category", "path", "text_field", "records_key"}:
            errors.append("invalid_overlap_source_schema")
            continue
        source_id = source["dataset_id"]
        if not isinstance(source_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,63}", source_id):
            errors.append("invalid_overlap_dataset_id")
            continue
        if source_id in dataset_ids:
            errors.append(f"duplicate_overlap_dataset_id:{source_id}")
            continue
        dataset_ids.add(source_id)
        if not all(isinstance(source[field], str) and source[field].strip()
                   for field in ("category", "path", "text_field")) or not isinstance(source["records_key"], str):
            errors.append(f"missing_overlap_source:{source_id}")
            continue
        path = (base / source["path"]).resolve()
        try:
            rows = read_rows(path, records_key=source["records_key"])
            if not rows:
                raise ValueError("Empty overlap source")
            texts = [row.get(source["text_field"]) for row in rows]
            if any(not isinstance(value, str) or not normalize_query(value) for value in texts):
                raise ValueError("Missing query text")
            unique_hashes = {digest(normalize_query(value)) for value in texts}
            for query_hash in unique_hashes:
                hashes[query_hash].append(source_id)
            categories.add(source["category"])
            evidence.append({"dataset_id": source_id, "category": source["category"],
                             "file_sha256": file_hash(path), "rows": len(rows),
                             "unique_normalized_queries": len(unique_hashes)})
        except (OSError, ValueError, KeyError, TypeError):
            errors.append(f"unreadable_or_invalid_overlap_source:{source_id}")
    errors.extend(f"unverified_overlap_category:{category}" for category in OVERLAP_CATEGORIES
                  if category not in categories)
    return hashes, evidence, errors


def audit_collection(plan: dict, queries: list[dict], participants: list[dict],
                     labels_a: list[dict], labels_b: list[dict], adjudications: list[dict],
                     known_hashes: dict, overlap_evidence: list[dict], overlap_errors: list[str]) -> dict:
    errors = list(overlap_errors)
    if (plan.get("version") != VERSION or plan.get("normalization") != NORMALIZATION
            or type(plan.get("target_per_intent")) is not int or plan["target_per_intent"] != TARGET_PER_INTENT):
        errors.append("collection_protocol_changed")
    if plan.get("protocol_reviewed") is not True:
        errors.append("protocol_review_pending")
    if plan.get("overlap_inventory_reviewed") is not True:
        errors.append("overlap_inventory_review_pending")
    batch_id = plan.get("batch_id")
    if not isinstance(batch_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,63}", batch_id):
        errors.append("invalid_collection_batch")

    def index_rows(rows: list[dict], field: str, name: str) -> dict:
        result = {}
        for row in rows:
            key = row[field]
            if key in result:
                errors.append(f"duplicate_{name}_id:{key}")
            else:
                result[key] = row
        return result

    people = index_rows(participants, "participant_id", "participant")
    for pid, person in people.items():
        if not re.fullmatch(r"p-[a-z0-9]{4,24}", pid):
            errors.append("invalid_participant_pseudonym")
        if not set(person["roles"].split("|")) <= ROLES:
            errors.append(f"invalid_participant_roles:{pid}")
        if any(person[key] not in {"true", "false"} for key in PARTICIPANT_FIELDS[2:]):
            errors.append(f"invalid_participant_attestation:{pid}")

    def has_role(pid: str, role: str) -> bool:
        return pid in people and role in people[pid]["roles"].split("|")

    if not has_role(plan.get("curator_id", ""), "collector"):
        errors.append("curator_missing_or_unregistered")
    query_map = index_rows(queries, "query_id", "query")
    reviews = [index_rows(rows, "query_id", name) for rows, name in
               ((labels_a, "labels_a"), (labels_b, "labels_b"), (adjudications, "adjudication"))]
    for name, mapping in zip(("labels_a", "labels_b", "adjudication"), reviews):
        errors.extend(f"unknown_{name}_query:{qid}" for qid in mapping if qid not in query_map)
    normalized_groups: dict[str, list[str]] = defaultdict(list)
    for qid, row in query_map.items():
        normalized_groups[digest(normalize_query(row["query"]))].append(qid)
    duplicates = {qid: min(group) for group in normalized_groups.values() for qid in group if qid != min(group)}
    source_ids = set()
    records, intent_counts = [], Counter({label: 0 for label in INTENTS})
    missing_labels, disagreements, missing_adjudications = 0, 0, 0
    for qid, row in query_map.items():
        flags, issues = [], []
        query_hash = digest(normalize_query(row["query"]))
        if not re.fullmatch(r"q-[a-z0-9_-]{1,60}", qid):
            issues.append("invalid_query_pseudonym")
        if not normalize_query(row["query"]):
            issues.append("empty_query")
        if not re.fullmatch(r"src-[a-z0-9_-]{1,60}", row["source_id"]):
            issues.append("invalid_source_pseudonym")
        if row["source_id"] in source_ids:
            issues.append("duplicate_source_id")
        source_ids.add(row["source_id"])
        try:
            collected = date.fromisoformat(row["collected_at"])
            if collected > date.today():
                issues.append("future_collection_date")
        except ValueError:
            issues.append("invalid_collection_date")
        if row["collection_batch"] != batch_id:
            issues.append("wrong_collection_batch")
        if row["provenance"] not in PROVENANCES:
            issues.append("invalid_provenance")
        author = row["author_id"]
        role = "writer" if row["provenance"] == "independent_human" else "collector"
        if not has_role(author, role):
            issues.append("author_missing_or_wrong_role")
        if row["provenance"] == "independent_human":
            if any(people.get(author, {}).get(field) != "true" for field in ("no_model_outputs", "no_old_queries")):
                issues.append("independent_author_attestation_missing")
        else:
            flags.append(row["provenance"])
            if not row["provenance_detail"].strip():
                issues.append("provenance_detail_missing")
        prior_sources = sorted(known_hashes.get(query_hash, []))
        if prior_sources:
            flags.append("known_overlap")
        if qid in duplicates:
            flags.append("normalized_duplicate")
        pair = [mapping.get(qid) for mapping in reviews[:2]]
        reviewer_ids = []
        for slot, annotation in enumerate(pair, 1):
            if not annotation or annotation["label"] not in LABELS:
                missing_labels += 1
                issues.append(f"missing_or_invalid_label_{slot}")
                continue
            rid = annotation["reviewer_id"]
            reviewer_ids.append(rid)
            if annotation["query"] != row["query"]:
                issues.append(f"reviewed_query_changed_{slot}")
            if not has_role(rid, "labeler"):
                issues.append(f"unregistered_labeler_{slot}")
            if any(people.get(rid, {}).get(field) != "true" for field in ("no_model_outputs", "independent_reviews")):
                issues.append(f"labeler_blinding_missing_{slot}")
            if rid == author:
                issues.append(f"author_is_labeler_{slot}")
        if len(reviewer_ids) == 2 and reviewer_ids[0] == reviewer_ids[1]:
            issues.append("labelers_not_independent")
        conflict = all(annotation and annotation["label"] in LABELS for annotation in pair) and pair[0]["label"] != pair[1]["label"]
        disagreements += int(conflict)
        adjudication = reviews[2].get(qid)
        final_label, disposition = None, "pending"
        if not adjudication:
            missing_adjudications += 1
            issues.append("missing_adjudication")
        else:
            aid = adjudication["adjudicator_id"]
            final_label = adjudication["final_label"]
            if final_label not in LABELS:
                issues.append("invalid_adjudicated_label")
            if not has_role(aid, "adjudicator") or people.get(aid, {}).get("no_model_outputs") != "true":
                issues.append("adjudicator_missing_or_not_blind")
            if aid in [author, *reviewer_ids]:
                issues.append("adjudicator_not_independent")
            if not adjudication["reason"].strip():
                issues.append("adjudication_reason_missing")
            for field in ("privacy_checked", "near_duplicate_checked"):
                if adjudication[field] != "true":
                    issues.append(f"{field}_pending")
            reason = adjudication["exclusion_reason"]
            near = adjudication["near_duplicate_of"]
            if near:
                # fresh:<qid> or old:<dataset_id>:<source pseudonym>; never raw query text.
                valid_near = (near.startswith("fresh:") and near[6:] in query_map and near[6:] != qid)
                if near.startswith("old:"):
                    parts = near.split(":")
                    valid_near = (len(parts) == 3 and parts[1] in {e["dataset_id"] for e in overlap_evidence}
                                  and bool(re.fullmatch(r"src-[a-z0-9_-]{1,60}", parts[2])))
                if not valid_near:
                    issues.append("invalid_near_duplicate_reference")
                flags.append("near_duplicate")
            needs_exclusion = bool(prior_sources or qid in duplicates or near)
            if adjudication["decision"] == "exclude":
                if reason not in {"duplicate", "near_duplicate", "privacy"}:
                    issues.append("invalid_exclusion_reason")
                elif reason == "duplicate" and not (prior_sources or qid in duplicates):
                    issues.append("unsubstantiated_duplicate_exclusion")
                elif reason == "near_duplicate" and not near:
                    issues.append("missing_near_duplicate_reference")
                disposition = "excluded"
                flags.append(f"excluded_{reason}")
            elif adjudication["decision"] == "retain":
                if needs_exclusion:
                    issues.append("duplicate_must_be_adjudicated_for_exclusion")
                if reason:
                    issues.append("retained_row_has_exclusion_reason")
                disposition = "supported" if final_label in INTENTS and row["provenance"] == "independent_human" else "secondary"
            else:
                issues.append("invalid_adjudication_decision")
            if final_label in {"ambiguous", "unsupported"}:
                flags.append(final_label)
        if issues:
            disposition = "pending"
        if disposition == "supported":
            intent_counts[final_label] += 1
        records.append({"query_id": qid, "source_id": row["source_id"], "author_id": author,
                        "normalized_query_sha256": query_hash, "final_label": final_label,
                        "disposition": disposition, "flags": sorted(set(flags)), "issues": issues,
                        "known_overlap_sources": prior_sources, "duplicate_of": duplicates.get(qid),
                        "label_disagreement": bool(conflict)})
    # Identical text with conflicting final labels must be resolved, even if excluded.
    for group in normalized_groups.values():
        final_labels = {reviews[2][qid]["final_label"] for qid in group if qid in reviews[2]}
        if len(final_labels) > 1:
            errors.append("conflicting_duplicate_labels:" + ",".join(sorted(group)))
    counts = Counter(record["disposition"] for record in records)
    deficits = {label: max(0, TARGET_PER_INTENT - intent_counts[label]) for label in INTENTS}
    if not records:
        errors.append("human_collection_pending")
    if counts["pending"]:
        errors.append("unfinished_records")
    if any(deficits.values()):
        errors.append("supported_target_not_met")
    ready = not errors
    return {
        "version": VERSION, "status": "ready_for_lock" if ready else "blocked",
        "collection_ready": ready, "evaluation_ready": False, "model_inference_performed": False,
        "target_per_intent": TARGET_PER_INTENT, "target_supported_total": 20 * TARGET_PER_INTENT,
        "counts": {"submitted": len(queries), "unique_query_ids": len(query_map),
                   **{key: counts[key] for key in ("supported", "secondary", "excluded", "pending")},
                   "missing_or_invalid_labels": missing_labels, "label_disagreements": disagreements,
                   "missing_adjudications": missing_adjudications,
                   "unique_writers": len({r["author_id"] for r in queries if r["provenance"] == "independent_human"})},
        "supported_per_intent": dict(intent_counts), "target_deficits": deficits,
        "flag_counts": dict(Counter(flag for record in records for flag in record["flags"])),
        "errors": sorted(set(errors)), "records": records,
        "overlap_sources": overlap_evidence,
        "collection_content_sha256": digest({"plan": plan, "queries": queries, "participants": participants,
                                               "labels_a": labels_a, "labels_b": labels_b, "adjudications": adjudications}),
        "limitations": "Attestations and supplied source inventory require human verification. Exact normalization does not prove semantic independence. No model or policy lock and no accuracy estimate exist here.",
    }


def ready_check(pack: Path, paths: dict[str, Path] | None = None) -> dict:
    paths = paths or {}
    plan = json.loads((pack / "collection-plan.json").read_text(encoding="utf-8-sig"))
    ontology = json.loads((pack / "ontology.json").read_text(encoding="utf-8-sig"))
    if ontology != ontology_document():
        raise ValueError("Ontology differs from the fixed collection protocol")
    rows = {name: read_rows(paths.get(name, pack / f"{name}.csv"), fields) for name, fields in
            (("queries", QUERY_FIELDS), ("participants", PARTICIPANT_FIELDS), ("labels-a", LABEL_FIELDS),
             ("labels-b", LABEL_FIELDS), ("adjudications", ADJUDICATION_FIELDS))}
    sources = json.loads((pack / "overlap-sources.json").read_text(encoding="utf-8-sig"))
    hashes, evidence, errors = overlap_index(sources, pack)
    return audit_collection(plan, rows["queries"], rows["participants"], rows["labels-a"], rows["labels-b"],
                            rows["adjudications"], hashes, evidence, errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    init = subparsers.add_parser("init", help="Create empty collection templates, never sample queries")
    init.add_argument("--output", type=Path, required=True)
    init.add_argument("--batch-id", required=True)
    labels = subparsers.add_parser("prepare-labels", help="Create two separate blinded label packets")
    labels.add_argument("--queries", type=Path, required=True)
    labels.add_argument("--output", type=Path, required=True)
    check = subparsers.add_parser("ready-check", help="Fail closed on missing data, labels, reviews or overlap checks")
    check.add_argument("--pack", type=Path, required=True)
    check.add_argument("--report", type=Path)
    for name in ("queries", "participants", "labels-a", "labels-b", "adjudications"):
        check.add_argument(f"--{name}", type=Path, help="Override CSV with CSV, JSON list/{records:[...]}, or JSONL")
    args = parser.parse_args(argv)
    try:
        if args.action == "init":
            output = private_output(args.output)
            create_pack(output, args.batch_id)
            print(json.dumps({"status": "human_collection_pending", "output": str(output), "queries_created": 0}))
        elif args.action == "prepare-labels":
            output = private_output(args.output)
            prepare_labels(args.queries, output)
            print(json.dumps({"status": "labeling_pending", "output": str(output)}))
        else:
            pack = private_output(args.pack)
            report_path = private_output(args.report or pack / "ready-report.json")
            inputs = {name: getattr(args, name.replace("-", "_")) for name in
                      ("queries", "participants", "labels-a", "labels-b", "adjudications")}
            paths = {name: path for name, path in inputs.items() if path is not None}
            protected = {path.resolve() for path in paths.values()}
            protected.update(path.resolve() for path in pack.glob("*")
                             if path.is_file() and path.name != "ready-report.json")
            sources_path = pack / "overlap-sources.json"
            try:
                sources = json.loads(sources_path.read_text(encoding="utf-8-sig"))
                if isinstance(sources, list):
                    protected.update((pack / source["path"]).resolve() for source in sources
                                     if isinstance(source, dict) and isinstance(source.get("path"), str)
                                     and source["path"])
            except (OSError, ValueError):
                pass  # ready_check reports a malformed or missing inventory below.
            if report_path in protected:
                raise ValueError("Report must not overwrite an input")
            if report_path.exists():
                old_report = json.loads(report_path.read_text(encoding="utf-8-sig"))
                if (not isinstance(old_report, dict) or old_report.get("version") != VERSION
                        or old_report.get("model_inference_performed") is not False
                        or old_report.get("status") not in {"blocked", "ready_for_lock"}):
                    raise ValueError("Report destination already contains another artifact")
            try:
                report = ready_check(pack, paths)
            except (OSError, ValueError, KeyError, TypeError) as error:
                report = {"version": VERSION, "status": "blocked", "collection_ready": False,
                          "evaluation_ready": False, "model_inference_performed": False,
                          "errors": [f"invalid_input:{type(error).__name__}"],
                          "detail": "An input is missing or violates the declared schema; no records were scored."}
            write_json(report_path, report)
            print(json.dumps({"status": report["status"], "counts": report.get("counts", {}),
                              "errors": report["errors"], "report": str(report_path), "evaluation_ready": False}))
            return 0 if report["collection_ready"] else 2
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Blocked: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
