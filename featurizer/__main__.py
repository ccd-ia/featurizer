# coding: utf-8

"""Allow running featurizer as a module: python -m featurizer"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
