# 4. ConvLSTM/GAT and structural refinement, V90
```bash
python -m pip install -r requirements.txt
python ../tools/materialize_assets.py --stage 04_ConvLSTM_GAT_V90
python run.py infer --output ../work/V90
```

```bash
python run.py train --output ../work/V90_retraining
```


