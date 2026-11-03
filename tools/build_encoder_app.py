"""Stage only application code for Flower's encoder deployment entry points."""
import argparse
from pathlib import Path
import shutil
import tomllib

ROOT=Path(__file__).resolve().parents[1]


def build(output):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError("Choose a fresh app bundle directory")
    text=(ROOT/"pyproject.toml").read_text()
    components=tomllib.loads(text)["tool"]["flwr"]["app"]["components"]
    expected={"serverapp":"fedlora_edge.flower_app:server_app",
              "clientapp":"fedlora_edge.flower_app:client_app"}
    if components != expected:
        raise ValueError("Review changed source Flower entry points before packaging")
    (output/"fedlora_edge").mkdir(parents=True)
    for path in (ROOT/"fedlora_edge").glob("*.py"):
        shutil.copy2(path,output/"fedlora_edge"/path.name)
    for name in ("README.md","LICENSE"):
        shutil.copy2(ROOT/name,output/name)
    text=text.replace("fedlora_edge.flower_app:server_app","fedlora_edge.encoder_deployment_app:server_app")
    text=text.replace("fedlora_edge.flower_app:client_app","fedlora_edge.encoder_deployment_app:client_app")
    (output/"pyproject.toml").write_text(text)
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(build(args.output.resolve()))
