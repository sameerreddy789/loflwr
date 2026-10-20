"""Check repository Markdown links and required project-skill frontmatter."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["README.md", "research/README.md", "research/landscape.md", "research/system-design.md", "research/evaluation.md", "research/roadmap.md", "research/sources.md"]
SKILLS = ["frozen-a-training", "federated-rounds", "banking77-experiments", "laptop-deployment"]


def main() -> int:
    errors = []
    for name in REQUIRED:
        if not (ROOT / name).is_file():
            errors.append(f"Missing required document: {name}")
    documents = []
    for folder in (ROOT, ROOT / "research", ROOT / "skills", ROOT / "tools"):
        documents.extend(folder.glob("*.md") if folder == ROOT else folder.rglob("*.md"))
    for path in sorted(set(documents)):
        text = path.read_text(encoding="utf-8-sig")
        body = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", body):
            target = target.strip().strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or target.startswith("#"):
                continue
            relative = unquote(parsed.path)
            if relative and not (path.parent / relative).exists():
                errors.append(f"{path.relative_to(ROOT)}: missing link target {target}")
    for name in SKILLS:
        path = ROOT / "skills" / name / "SKILL.md"
        if not path.is_file():
            errors.append(f"Missing skill: {name}")
            continue
        text = path.read_text(encoding="utf-8-sig")
        if not text.startswith("---\n") or f"name: {name}\n" not in text or not re.search(r"^description:\s*\S", text, re.MULTILINE):
            errors.append(f"Invalid required skill frontmatter: {name}")
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Checked {len(set(documents))} Markdown files, local file links, and {len(SKILLS)} project skills.")
    print("External link availability and Markdown anchors require separate source/render review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
