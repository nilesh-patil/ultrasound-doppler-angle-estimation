# legacy/ — original 2017–2018 implementation (archived, untouched)

These are the **original artifacts** behind the EMBC 2019 paper, kept verbatim for
provenance and sanity comparison. **Do not extend or edit them** — the v2 library
under `src/uda/` is a clean re-implementation.

| Path | What it is |
|---|---|
| `notebooks/` | the 9 original Jupyter notebooks (Keras 2.x / TF1 / Py2 idioms) |

Reverse-engineered pipeline (from the notebooks): standalone Keras 2.x on TF1,
`fit_generator`, grayscale 100×100 custom CNNs **and** the transfer-learning path
(backbones → `deepdish` HDF5 feature cache → shallow head, with an MC-dropout
"new" generation). The paper reports the transfer-learning path.
