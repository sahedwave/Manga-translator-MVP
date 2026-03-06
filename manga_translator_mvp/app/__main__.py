import sys

from .cli import main as cli_main
from .gui import main as web_main


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli_main()
    else:
        web_main()
