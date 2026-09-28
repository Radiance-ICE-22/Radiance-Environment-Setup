import sys

import uvicorn

from .app import create_app
from .settings import NoMachineProfile, load


def main() -> None:
    try:
        s = load()
    except NoMachineProfile as e:
        sys.exit(f"galley: {e}")
    print(f"galley: profile {s.source}, serving on {s.host}:{s.port}", flush=True)
    uvicorn.run(create_app(s), host=s.host, port=s.port, log_level="info")


if __name__ == "__main__":
    main()
