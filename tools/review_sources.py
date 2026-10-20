"""Record reproducible, bounded primary-page checks through the Jina Reader route.

This checks metadata and abstracts, not experimental reproducibility. Raw reader
responses stay in a temporary directory rather than the Git repository.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


SOURCES = {
    "lora": "https://arxiv.org/abs/2106.09685",
    "qlora": "https://arxiv.org/abs/2305.14314",
    "ffa-lora": "https://arxiv.org/abs/2403.12313",
    "lora-fa": "https://arxiv.org/abs/2308.03303",
    "dora": "https://arxiv.org/abs/2402.09353",
    "adalora": "https://arxiv.org/abs/2303.10512",
    "vera": "https://arxiv.org/abs/2310.11454",
    "galore": "https://arxiv.org/abs/2403.03507",
    "mezo": "https://arxiv.org/abs/2305.17333",
    "fedavg": "https://arxiv.org/abs/1602.05629",
    "fedprox": "https://arxiv.org/abs/1812.06127",
    "scaffold": "https://arxiv.org/abs/1910.06378",
    "fednova": "https://arxiv.org/abs/2007.07481",
    "q-ffl": "https://arxiv.org/abs/1905.10497",
    "fedbuff": "https://arxiv.org/abs/2106.06639",
    "flora": "https://arxiv.org/abs/2409.05976",
    "flexlora": "https://arxiv.org/abs/2402.11505",
    "hetlora": "https://arxiv.org/abs/2401.06432",
    "prelort-plan-citation": "https://arxiv.org/abs/2606.15963",
    "fedlore-plan-citation": "https://arxiv.org/abs/2610.01620",
    "fedfit-plan-citation": "https://arxiv.org/abs/2610.01537",
    "update-leakage": "https://arxiv.org/abs/1906.08935",
    "secure-aggregation": "https://research.google/pubs/practical-secure-aggregation-for-privacy-preserving-machine-learning/",
    "banking77": "https://github.com/PolyAI-LDN/task-specific-datasets",
    "smollm2-135m": "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct",
    "smollm2-360m": "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct",
    "gpt2": "https://huggingface.co/openai-community/gpt2",
    "distilgpt2": "https://huggingface.co/distilbert/distilgpt2",
    "gemma3-270m": "https://huggingface.co/google/gemma-3-270m-it",
    "qwen2.5-0.5b": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct",
    "lfm2-350m": "https://huggingface.co/LiquidAI/LFM2-350M",
    "tinyllama": "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "peft": "https://huggingface.co/docs/peft/main/developer_guides/lora",
    "flower-simulation": "https://flower.ai/docs/framework/how-to-run-simulations.html",
    "flower-grid": "https://flower.ai/docs/framework/ref-api/flwr.serverapp.Grid.html",
    "flowertune": "https://flower.ai/docs/examples/flowertune-llm.html",
    "federatedscope-llm": "https://github.com/alibaba/FederatedScope",
    "openfedllm": "https://github.com/rui-ye/OpenFedLLM",
    "flute": "https://github.com/microsoft/msrflute",
    "fedllm-bench": "https://github.com/rui-ye/FedLLM-Bench",
}


def review(label: str, url: str, cache: Path, date: str, timeout: float) -> dict:
    record = {"label": label, "url": url, "access_date": date, "review_depth": "metadata/abstract or documentation page; no reproduction"}
    reader_url = "https://r.jina.ai/" + url
    request = urllib.request.Request(reader_url, headers={"User-Agent": "FedLoRA-Edge-source-review/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(250_000).decode("utf-8", errors="replace")
            record["reader_http_status"] = response.status
        (cache / (hashlib.sha256(url.encode()).hexdigest()[:16] + ".md")).write_text(body, encoding="utf-8")
        title = re.search(r"^Title:\s*(.+)$", body, re.MULTILINE)
        comments = re.search(r"^Comments:\s*(.+)$", body, re.MULTILINE)
        record["title"] = title.group(1).strip() if title else None
        record["publication_comments"] = comments.group(1).strip() if comments else None
        record["status"] = "page-read" if title and "not found" not in title.group(1).casefold() else "needs-manual-review"
        record["license_links"] = sorted(set(re.findall(r"https?://(?:creativecommons\.org/licenses/[^\s)\]]+|www\.apache\.org/licenses/[^\s)\]]+)", body)))
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        record.update(status="unavailable", error=str(error))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Access date in the researcher's timezone, YYYY-MM-DD")
    parser.add_argument("--output", type=Path, default=Path("research/source-review.json"))
    parser.add_argument("--timeout", type=float, default=35)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--labels", nargs="*", choices=sorted(SOURCES))
    args = parser.parse_args()
    cache = Path(tempfile.mkdtemp(prefix="fedlora-source-review-"))
    labels = args.labels or list(SOURCES)
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(review, name, SOURCES[name], cache, args.date, args.timeout): name for name in labels}
        for result in concurrent.futures.as_completed(pending):
            record = result.result()
            records.append(record)
            print(json.dumps({key: record.get(key) for key in ("label", "status", "title", "publication_comments", "error")}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
    combined = {record["label"]: record for record in previous.get("records", [])}
    combined.update({record["label"]: record for record in records})
    previous.update(access_date=args.date, route="agent-reach/Jina Reader", records=sorted(combined.values(), key=lambda x: x["label"]))
    args.output.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
    print(f"Metadata: {args.output}; temporary reader cache: {cache}")


if __name__ == "__main__":
    main()
