from pathlib import Path


def rm_tree(dir: Path) -> None:
    """
    Removes recursively all files and directories within the given directory.
    """
    for child in dir.iterdir():
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    dir.rmdir()
