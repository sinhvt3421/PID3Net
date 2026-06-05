# About

## Authors

PID3Net is developed by **Vu Tien-Sinh** and collaborators at the
Japan Advanced Institute of Science and Technology (JAIST).

- Email: <sinh.vt@jaist.ac.jp>
- GitHub: [sinhvt3421](https://github.com/sinhvt3421)

## License

PID3Net is released under the [MIT License](https://opensource.org/licenses/MIT).
You are free to use, modify, and redistribute it, including for
commercial purposes, subject to the license terms.

```
Copyright (c) Vu Tien-Sinh and contributors

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files...
```

(Full text in `LICENSE` at the repo root.)

## Citation

If you use PID3Net in academic work, please cite:

```bibtex
@software{vu2025pid3net,
  author = {Vu, Tien-Sinh and contributors},
  title  = {PID3Net: Physics-Informed Deep learning Network for Dynamic Diffraction imaging},
  year   = {2025},
  url    = {https://github.com/sinhvt3421/PID3Net},
  version = {2.0.0},
}
```

A companion publication is in preparation; this entry will be updated
with the journal reference upon publication.

## Acknowledgements

The model builds on ideas from:

- **ePIE / RAAR** (Maiden & Rodenburg 2009; Luke 2005) — the
  iterative-projection algorithms the refinement block mirrors.
- **Maximum-likelihood ptychography** (Thibault & Guizar-Sicairos 2012) —
  the Poisson-MLE formulation underlying the optional Poisson projection.
- Earlier deep-learning ptychography work — notably AutoPhaseNN (Yao
  *et al.*) and PtychoNN (Cherukara *et al.*) — for the
  encoder–decoder skeleton that PID3Net extends with a 3D temporal
  axis and a learned refinement block.

## Project links

- **Source**: <https://github.com/sinhvt3421/PID3Net>
- **Issue tracker**: <https://github.com/sinhvt3421/PID3Net/issues>
- **Documentation**: <https://sinhvt3421.github.io/pid3net/>
- **Changelog**: [changelog.md](changelog.md)
