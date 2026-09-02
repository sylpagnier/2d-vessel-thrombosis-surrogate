# Legacy research sweep configs

Configs that target the retired mat-growth / biochem deploy stack (`locked_canonical`).

Run with:

```powershell
powershell ... -File .\scripts\go_research_sweep.ps1 -Sweep 15_stack_coupling -Legacy
```

Active geometry sweeps (01-14, 16-20) use `clot_ml_0` + FEM t=0 flow by default.
