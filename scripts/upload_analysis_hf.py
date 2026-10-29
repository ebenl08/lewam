"""Upload the LeWAM and LeWM analysis outputs (plots, tables, notes) to a private Hugging Face repo in one commit.

Repo layout:
- lewam/...  every allowed file under playground/lewam/data/**/analysis/, data/trm/probe/ and the plots at the top
  of data/<experiment>/ (path relative to playground/lewam/data/)
- lewm/...   plots, tables and summaries under playground/stable-worldmodel/data/experiment/ (path relative to it;
  per-run config.json files and the _raw_ folder are left out)
- notes/lewam/, notes/stable-worldmodel/  with --include-notes (Markdown; links to local paths will not resolve)
The repo is created private if missing; an existing public repo is refused. Auth: `huggingface-cli login` or
HF_TOKEN. Run from playground/lewam:
    /home/eben/.conda/envs/lewam/bin/python scripts/upload_analysis_hf.py --repo-id <user>/<repo> --dry-run
"""

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

LW = Path(__file__).resolve().parents[1]
REPO_ROOT = LW.parents[1]
LEWAM_DATA = LW / "data"
LEWM_DATA = REPO_ROOT / "playground" / "stable-worldmodel" / "data" / "experiment"
NOTES = {"notes/lewam": REPO_ROOT / "notes" / "lewam", "notes/stable-worldmodel": REPO_ROOT / "notes" / "stable-worldmodel"}
LEWAM_EXTENSIONS = {".png", ".pdf", ".svg", ".csv", ".json", ".md"}
LEWM_EXTENSIONS = {".png", ".pdf", ".svg", ".csv", ".md"}     # no .json: about 1,200 per-run configs


def lewam_files() -> dict[str, Path]:
    files = {}
    for root in [*LEWAM_DATA.glob("**/analysis"), LEWAM_DATA / "trm" / "probe"]:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in LEWAM_EXTENSIONS:
                files[f"lewam/{path.relative_to(LEWAM_DATA)}"] = path
    for path in LEWAM_DATA.glob("*/*.png"):                        # top-level experiment plots, e.g. SR heatmaps
        files[f"lewam/{path.relative_to(LEWAM_DATA)}"] = path
    return files


def lewm_files() -> dict[str, Path]:
    files = {}
    for path in LEWM_DATA.rglob("*"):
        relative = path.relative_to(LEWM_DATA)
        if relative.parts[0] != "_raw_" and path.is_file() and path.suffix in LEWM_EXTENSIONS:
            files[f"lewm/{relative}"] = path
    return files


def note_files() -> dict[str, Path]:
    return {f"{prefix}/{path.relative_to(root)}": path
            for prefix, root in NOTES.items() for path in root.rglob("*.md")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-id", required=True, help="<user or org>/<repo name>")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    parser.add_argument("--skip-lewm", action="store_true", help="LeWAM outputs only")
    parser.add_argument("--include-notes", action="store_true", help="also upload notes/lewam and notes/stable-worldmodel")
    parser.add_argument("--message", default="Upload LeWAM and LeWM analysis outputs")
    parser.add_argument("--dry-run", action="store_true", help="list the files and sizes, upload nothing")
    args = parser.parse_args()

    files = lewam_files()
    if not args.skip_lewm:
        files.update(lewm_files())
    if args.include_notes:
        files.update(note_files())
    files = dict(sorted(files.items()))
    total_mb = sum(path.stat().st_size for path in files.values()) / 1e6
    for repo_path in files:
        print(repo_path)
    print(f"{len(files)} files, {total_mb:.1f} MB -> {args.repo_type} repo {args.repo_id} (private)")
    if args.dry_run:
        return

    api = HfApi()
    api.create_repo(args.repo_id, repo_type=args.repo_type, private=True, exist_ok=True)
    if not api.repo_info(args.repo_id, repo_type=args.repo_type).private:
        raise SystemExit(f"{args.repo_id} exists and is public; refusing to upload")
    operations = [CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(path))
                  for repo_path, path in files.items()]
    commit = api.create_commit(args.repo_id, operations, commit_message=args.message, repo_type=args.repo_type)
    print(f"uploaded: {commit.commit_url}")


if __name__ == "__main__":
    main()
