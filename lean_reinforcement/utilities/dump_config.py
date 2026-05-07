import json
import sys
from dataclasses import asdict

from lean_reinforcement.utilities.config import get_config


def main() -> None:
    config = get_config()
    json.dump(asdict(config), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
