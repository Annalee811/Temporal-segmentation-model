# 1. Adjacent matching and registered observations

The matcher is the full upstream RoMa outdoor network with DINOv2 ViT-L/14.
Its source and original license are retained under `vendor`. The supplied
four-period example uses one frozen similarity per observation region;
the RGB images are never nonrigidly warped by the dense correspondence field.

```bash
python -m pip install -r requirements.txt
python ../tools/materialize_assets.py --stage 01_adjacent_matching
python run.py prepare --raw-root assets/raw --work ../work/registration
python run.py render --work ../work/registration
python confidence_cached.py --work ../work/registration
```

`render` is an explicitly frozen-geometry replay, not new pose estimation.
Its four RGB outputs are SHA-256 checked and are pixel-identical to the RGB
arrays used by stages 2 and 3. `confidence_cached.py` recomputes cumulative
confidence and overlap from supplied original RoMa fields; it does not claim
to rerun RoMa. The original neural inference command is:

```bash
python run.py confidence --work ../work/registration
```

This requires the two official weights listed with complete download URLs
and SHA-256 values in `assets/weight_downloads.json`. Their combined size is
about 1.6 GB. The upstream loader downloads to the standard Torch cache and
verifies the pinned hashes. Resolution is560/864, symmetric matching is enabled,
the seed is20261019, and custom CUDA correlation is disabled. CPU inference is
supported but slower. Fresh neural matching was not rerun during preparation
of this submission snapshot; geometric reconstruction and cached-field quality
calculations were rerun separately.

Cumulative confidence transports coordinates through adjacent observed dates
and multiplies the smaller forward/reverse score at each step. Effective
support requires confidence≥0.5, step displacement≤15 and cycle error≤10 native
reference pixels. These are operational matching criteria, not independently
measured landmark accuracy. Missing periods must not be synthesized.
