from utils.logger import log


def todos() -> None:
    log.info("Add auto documentation generation.")


def main() -> None:
    log.info("Hello from python-template!")
    todos()


if __name__ == "__main__":
    main()
