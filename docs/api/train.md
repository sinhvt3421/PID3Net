# `pid3net.train`

Console-script entry point and pipeline helpers. Invoked as
`pid3net-train` (installed script), `python -m pid3net.train`, or
through the back-compat shim at `train_ssp.py`.

::: pid3net.train
    options:
      heading_level: 3
      show_root_heading: false
      members:
        - main
        - run
        - build_parser
        - setup_gpu
        - set_seed
        - load_config
        - load_model
