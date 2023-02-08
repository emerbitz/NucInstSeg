from pathlib import Path
from typing import List


def comp_dirs(dir: Path, comp_dir: Path) -> List[Path]:
    """
    Returns the files that are in dir but not in comp_dir
    """

    files = {f for f in dir.iterdir() if f.is_file()}
    comp_files = {f.stem for f in comp_dir.iterdir() if f.is_file()}
    missing_files = []
    for file in files:
        if file.stem not in comp_files:
            missing_files.append(file)
    return missing_files
