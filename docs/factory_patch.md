# Registering TartanDriveDataSource in the factory

The repo's data source factory (`src/wm_datasets/data_source/factory.py`)
dispatches on dataset name. You need to add a branch for `tartandrive`.

## What it probably looks like

Open `src/wm_datasets/data_source/factory.py`. You'll see something like:

```python
def build_data_source(name: str, **kwargs):
    if name == "csgo":
        from src.wm_datasets.data_source.game.csgo import CSGODataSource
        return CSGODataSource(**kwargs)
    elif name.startswith("dino_wm"):
        from src.wm_datasets.data_source.dino_wm import DinoWMDataSource
        return DinoWMDataSource(name=name, **kwargs)
    elif name == "rt1":
        ...
    else:
        raise ValueError(f"Unknown dataset: {name}")
```

## What to add

Add a branch for `tartandrive`:

```python
elif name == "tartandrive":
    from src.wm_datasets.data_source.offroad.tartandrive import TartanDriveDataSource
    return TartanDriveDataSource(**kwargs)
```

The exact dispatch syntax (if/elif chain, dict lookup, registry decorator,
etc.) depends on what's already there. Match the existing pattern.

## Verifying it worked

After registering, run the smoke test (`scripts/smoke_test.sh`). If it errors
with `Unknown dataset: tartandrive`, the registration didn't take. If it
errors somewhere inside `TartanDriveDataSource.__init__`, the registration
worked but the DataSource has a bug — most commonly a wrong path in the
config or a base-class signature mismatch.
