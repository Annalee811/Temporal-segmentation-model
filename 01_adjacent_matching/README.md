# 1. Adjacent matching and registered observations

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



