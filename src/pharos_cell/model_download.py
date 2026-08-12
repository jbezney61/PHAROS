"""Download and verify the pretrained model artifacts used by PHAROS."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path
from typing import Mapping, Sequence


ST_REPO_ID = "arcinstitute/ST-SE-Tahoe"
ST_REVISION = "03b1971d7cc93a7535fd2e957c6948dba267378b"
ST_RUN_RELATIVE = Path("fewshot/state_generalization_X_state")
ST_CHECKPOINT_RELATIVE = ST_RUN_RELATIVE / "checkpoints/final.ckpt"

SE_REPO_ID = "arcinstitute/SE-600M"
SE_REVISION = "5a9a80f44f7ce32ce57059933ef0d735d7c10ce5"
SE_CHECKPOINT_RELATIVE = Path("se600m_epoch16.ckpt")

ST_CHECKSUMS: Mapping[str, str] = {
    "fewshot/state_generalization_X_state/batch_onehot_map.pkl": "51aa4f2ba68805ebe305f77338cece6e0898e7f53da23519636504f076c53f07",
    "fewshot/state_generalization_X_state/cell_type_onehot_map.pkl": "4edff99697df91d039d2e1679d4d5a2de9d173b06efb14516be33ce1cc29b098",
    "fewshot/state_generalization_X_state/checkpoints/final.ckpt": "7fc49ead44ea82267c5dfc4ab6f00ed04a4b0afff55175d13fa3f1ca357196c2",
    "fewshot/state_generalization_X_state/config.yaml": "30fcec6e52e73bbbbc0b843ba9c0e749fb4a85356dbe78cff58a7477f2ca65e4",
    "fewshot/state_generalization_X_state/data_module.torch": "d896b39d55fba7fa3b6c149d88ca5dea71317582ffccce67d71459cdf559df2a",
    "fewshot/state_generalization_X_state/pert_onehot_map.pt": "3eeb67ceefea8677ee6f0d48d0fd7a5022742847cbfefe461e453e715106181d",
    "fewshot/state_generalization_X_state/var_dims.pkl": "be09a1c4d7338c340ecd2193d3cf6ce691286645775c6b4c139a81d1cc4d3f3e",
    "fewshot/state_generalization_X_state/version_0/hparams.yaml": "4bea7dae5f5d035ce194a635411fe0bd195e7098a7ce9c8420f97afac38bd9b0",
    "fewshot/state_generalization_X_state/wandb_path.txt": "baa42d8f34b15a2c722a7ba2cb1781e8433cf620ada5072f727402e054cec31c",
}

SE_CHECKSUMS: Mapping[str, str] = {
    ".gitattributes": "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    "LICENSE.md": "e66c269d4819aaab34b49ef5220c4ddab6756f21bb5180761a4eb8561f2b7bbd",
    "MODEL_ACCEPTABLE_USE_POLICY.md": "6eb675a79981d59923be82e3931ef45187b373d7dff04ec7497c897d17d35c51",
    "MODEL_LICENSE.md": "513add0eb2b35e59ce7cd7ff1249b9104794f5fb879ec287a4a9a52e4a96b7a5",
    "README.md": "739f9dc508639ab84802f718ec495d2bcd30811b3510cfe49a2bb534c0eacde1",
    "config.yaml": "807d9f741a7b3c1243f361fa5672c92b3d7f775fccf890136cd8bff41618881a",
    "model.safetensors": "3ebec58bd9e9c07f0a76de25b900dad13d015768fbff1e68e5dcbdfb3b2245de",
    "protein_embeddings.pt": "a210e1cc7901513999b2bca3836ba9e2f203cd008be4e9a9d6412a2267de9748",
    "se600m_epoch16.ckpt": "b49bab144471f3b9318e1a661fb7b78bfa5110b500aee5c89d660f5f5927b7a5",
    "se600m_epoch4.ckpt": "736d60e47dd47aa5059ee3deda7662e9bfd2cf23200a899a65d49a294fc8c0f9",
    "se600m_epoch4.safetensors": "9d0cc6fc5aa89ed27c6c2ae336aa304ada4357518e40ad95351388e671d452e1",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pharos models download",
        description="Download the pinned SE-600M and ST-SE-Tahoe artifacts used by PHAROS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Parent directory for ST-SE-Tahoe/ and SE-600M/.",
    )
    parser.add_argument(
        "--component",
        choices=["all", "transition", "embedding"],
        default="all",
        help="Download both models, only ST-SE-Tahoe, or only SE-600M.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload files even if Hugging Face considers the local copies current.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not access Hugging Face; use only files already available locally or in cache.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum concurrent Hugging Face file downloads.",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Skip SHA-256 verification (not recommended).",
    )
    parser.set_defaults(verify=True)
    return parser


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(root: Path, checksums: Mapping[str, str]) -> None:
    problems: list[str] = []
    for relative, expected in checksums.items():
        path = root / relative
        if not path.is_file():
            problems.append(f"missing: {path}")
            continue
        observed = _sha256(path)
        if observed != expected:
            problems.append(
                f"checksum mismatch: {path}\n"
                f"  expected: {expected}\n"
                f"  observed: {observed}"
            )
    if problems:
        raise RuntimeError("Model verification failed:\n" + "\n".join(problems))


def _snapshot_download(
    *,
    repo_id: str,
    revision: str,
    local_dir: Path,
    allow_patterns: Sequence[str],
    force_download: bool,
    local_files_only: bool,
    max_workers: int,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Downloading PHAROS models requires huggingface_hub in the active environment."
        ) from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir),
        allow_patterns=list(allow_patterns),
        force_download=force_download,
        local_files_only=local_files_only,
        max_workers=max_workers,
    )


def model_paths(output_dir: Path) -> dict[str, Path]:
    root = output_dir.expanduser().resolve()
    st_dir = root / "ST-SE-Tahoe"
    se_dir = root / "SE-600M"
    return {
        "root": root,
        "st_dir": st_dir,
        "st_run": st_dir / ST_RUN_RELATIVE,
        "st_checkpoint": st_dir / ST_CHECKPOINT_RELATIVE,
        "se_dir": se_dir,
        "se_checkpoint": se_dir / SE_CHECKPOINT_RELATIVE,
    }


def environment_lines(output_dir: Path) -> list[str]:
    paths = model_paths(output_dir)
    return [
        f"export ST_RUN={shlex.quote(str(paths['st_run']))}",
        f"export ST_CKPT={shlex.quote(str(paths['st_checkpoint']))}",
        f"export SE_DIR={shlex.quote(str(paths['se_dir']))}",
        f"export SE_CKPT={shlex.quote(str(paths['se_checkpoint']))}",
    ]


def write_manifest(output_dir: Path, component: str, verified: bool) -> Path:
    paths = model_paths(output_dir)
    payload = {
        "format_version": 1,
        "component": component,
        "verified": bool(verified),
        "models": {
            "transition": {
                "repo_id": ST_REPO_ID,
                "revision": ST_REVISION,
                "directory": str(paths["st_dir"]),
                "run_directory": str(paths["st_run"]),
                "checkpoint": str(paths["st_checkpoint"]),
                "checkpoint_sha256": ST_CHECKSUMS[str(ST_CHECKPOINT_RELATIVE)],
            },
            "embedding": {
                "repo_id": SE_REPO_ID,
                "revision": SE_REVISION,
                "directory": str(paths["se_dir"]),
                "checkpoint": str(paths["se_checkpoint"]),
                "checkpoint_sha256": SE_CHECKSUMS[str(SE_CHECKPOINT_RELATIVE)],
            },
        },
    }
    path = paths["root"] / "pharos_model_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_workers < 1:
        raise SystemExit("error: --max-workers must be at least 1")

    paths = model_paths(args.output_dir)
    if args.component in {"all", "transition"}:
        print(f"Downloading {ST_REPO_ID} at revision {ST_REVISION}")
        _snapshot_download(
            repo_id=ST_REPO_ID,
            revision=ST_REVISION,
            local_dir=paths["st_dir"],
            allow_patterns=tuple(ST_CHECKSUMS),
            force_download=args.force_download,
            local_files_only=args.local_files_only,
            max_workers=args.max_workers,
        )
        if args.verify:
            print("Verifying ST-SE-Tahoe SHA-256 checksums")
            verify_files(paths["st_dir"], ST_CHECKSUMS)

    if args.component in {"all", "embedding"}:
        print(f"Downloading {SE_REPO_ID} at revision {SE_REVISION}")
        _snapshot_download(
            repo_id=SE_REPO_ID,
            revision=SE_REVISION,
            local_dir=paths["se_dir"],
            allow_patterns=tuple(SE_CHECKSUMS),
            force_download=args.force_download,
            local_files_only=args.local_files_only,
            max_workers=args.max_workers,
        )
        if args.verify:
            print("Verifying SE-600M SHA-256 checksums")
            verify_files(paths["se_dir"], SE_CHECKSUMS)

    manifest = write_manifest(paths["root"], args.component, args.verify)
    print(f"Model manifest: {manifest}")
    print("\nUse these paths in the current shell:")
    print("\n".join(environment_lines(paths["root"])))


if __name__ == "__main__":
    main()
