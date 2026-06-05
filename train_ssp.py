"""Backward-compatible shim — the real entry point is ``pid3net.train.main``.

Kept so existing invocations like ``python train_ssp.py configs/foo.yaml`` keep
working.  New code should use the installed console script ``pid3net-train``
or ``python -m pid3net.train``.
"""

from pid3net.train import main

if __name__ == "__main__":
    main()
