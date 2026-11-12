"""Count, rename, and optionally duplicate images in the signer dataset.

Expected layout::

    data/<signer>/<sign>/<image>

The script previews changes by default. Pass ``--apply`` to rename images and
copy existing images until each non-empty sign folder reaches the target count.
"""

from __future__ import annotations

import argparse
import shutil
import uuid
from pathlib import Path


DEFAULT_SIGNERS = ("Areeb", "Ayomide", "Chirayu")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def image_files(folder: Path) -> list[Path]:
    """Return image files directly inside a sign folder in stable order."""
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.casefold(),
    )


def count_dataset(root: Path, signers: tuple[str, ...]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for signer in signers:
        signer_dir = root / signer
        if not signer_dir.is_dir():
            print(f"WARNING: signer folder not found: {signer_dir}")
            continue
        for sign_dir in sorted((path for path in signer_dir.iterdir() if path.is_dir()), key=lambda p: p.name.casefold()):
            count = len(image_files(sign_dir))
            counts[(signer, sign_dir.name)] = count
            print(f"{signer:12} {sign_dir.name:8} {count:3} images")
    return counts


def plan_folder(sign_dir: Path, target: int) -> tuple[list[tuple[Path, str]], list[tuple[Path, Path]]]:
    """Plan normalized names and copies for one sign folder."""
    originals = image_files(sign_dir)
    rename_plan = [
        (source, f"{sign_dir.name}_{index:03d}{source.suffix.lower()}")
        for index, source in enumerate(originals, start=1)
    ]
    copy_plan: list[tuple[Path, Path]] = []
    if originals and len(originals) < target:
        for index in range(len(originals), target):
            source = originals[index % len(originals)]
            destination = sign_dir / f"{sign_dir.name}_{index + 1:03d}{source.suffix.lower()}"
            copy_plan.append((source, destination))
    return rename_plan, copy_plan


def apply_folder(sign_dir: Path, target: int) -> None:
    """Rename without clobbering collisions, then copy up to target count."""
    rename_plan, copy_plan = plan_folder(sign_dir, target)
    original_paths = {source for source, _ in rename_plan}
    final_paths = {sign_dir / name for _, name in rename_plan}
    conflicts = [path for path in final_paths if path.exists() and path not in original_paths]
    if conflicts:
        raise FileExistsError(f"Refusing to overwrite existing file: {conflicts[0]}")

    # Move every source to a unique temporary name first so pre-existing names
    # cannot collide with the normalized names during a rename.
    temporary: list[tuple[Path, Path]] = []
    for source, final_name in rename_plan:
        temp_path = sign_dir / f".rename-{uuid.uuid4().hex}{source.suffix.lower()}"
        source.rename(temp_path)
        temporary.append((temp_path, sign_dir / final_name))

    renamed_sources = {source: sign_dir / name for source, name in rename_plan}
    for temp_path, destination in temporary:
        temp_path.rename(destination)

    for source, destination in copy_plan:
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
        shutil.copy2(renamed_sources[source], destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count images by signer and sign, rename them, and fill non-empty folders to a target count."
    )
    parser.add_argument("--root", type=Path, default=Path("data"), help="Dataset root (default: data)")
    parser.add_argument("--target", type=int, default=20, help="Images per sign folder (default: 20)")
    parser.add_argument(
        "--signers",
        nargs="+",
        default=list(DEFAULT_SIGNERS),
        help="Signer folder names (default: Areeb Ayomide Chirayu)",
    )
    parser.add_argument("--apply", action="store_true", help="Make changes; without this flag, only preview them")
    args = parser.parse_args()

    root = args.root.resolve()
    if args.target < 1:
        parser.error("--target must be at least 1")
    if not root.is_dir():
        parser.error(f"dataset root does not exist or is not a directory: {root}")

    print(f"Dataset root: {root}")
    print("Current image counts:")
    count_dataset(root, tuple(args.signers))

    action = "APPLYING" if args.apply else "DRY RUN (no files will be changed)"
    print(f"\n{action}: target is {args.target} images per non-empty sign folder")
    for signer in args.signers:
        signer_dir = root / signer
        if not signer_dir.is_dir():
            continue
        for sign_dir in sorted((path for path in signer_dir.iterdir() if path.is_dir()), key=lambda p: p.name.casefold()):
            rename_plan, copy_plan = plan_folder(sign_dir, args.target)
            if not rename_plan:
                print(f"SKIP {signer}/{sign_dir.name}: no images to copy")
                continue
            print(f"{signer}/{sign_dir.name}: rename {len(rename_plan)}, add {len(copy_plan)}")
            if args.apply:
                apply_folder(sign_dir, args.target)

    if args.apply:
        print("\nImage counts after changes:")
        count_dataset(root, tuple(args.signers))
    else:
        print("\nPreview only. Re-run with --apply to make these changes.")


if __name__ == "__main__":
    main()
