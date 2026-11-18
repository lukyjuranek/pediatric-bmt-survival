"""Run the Lab II analysis and write figures, tables, and results.json."""

from bmt.data import default_data_path, repo_root
from bmt.pipeline import run_analysis


def main() -> None:
    run_analysis(default_data_path(), repo_root())


if __name__ == "__main__":
    main()
