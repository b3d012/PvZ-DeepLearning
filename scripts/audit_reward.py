from pvz_deeplearning.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["audit-reward", *__import__("sys").argv[1:]]))
