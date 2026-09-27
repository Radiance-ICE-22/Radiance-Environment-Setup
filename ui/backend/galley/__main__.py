import uvicorn

from .app import create_app
from .settings import load


def main() -> None:
    s = load()
    uvicorn.run(create_app(s), host=s.host, port=s.port, log_level="info")


if __name__ == "__main__":
    main()
