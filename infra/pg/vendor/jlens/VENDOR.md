# Vendored: anthropics/jacobian-lens @ 581d398613e5602a5af361e1c34d3a92ea82ba8e (2026-07-02), Apache-2.0

Files copied verbatim from `jlens/` at that commit (fetched 2026-09-04 from raw.githubusercontent.com); `vis.py`,
`examples.py` and `data/` are not needed and not copied. sha256:

| file | sha256 |
|---|---|
| `__init__.py` | bd7bb6302565a712583b05b91d48c8638d9f5d2ace00e52db0db2fd300f7fe3a |
| `_logging.py` | 5943d28520b8fe0e1b2f8ba16d7802522a2370f11f213bc6bb694e09bcfc06ae |
| `fitting.py` | 5be8959db8efc34cee41ed677beba84e21ba3c9e3ccb958bdbc1600c86b5e080 |
| `hf.py` | 228cf078e4586a7b7f61a6f5064403b8960de337afd19256efa56f04d53e3222 |
| `hooks.py` | c781d6944fd23396d3fc65a04db1f1db807f6f12cd5912cdbd2fb67eb3508081 |
| `lens.py` | e231e7d3a6c8e8f7791b53705a34342d0bba376a127a82376eaf6ec30ca11808 |
| `protocol.py` | 7217a3bc2c1cd01fed3e6b00c53effbd957effddc49fe073da89cb10d1b1a344 |
| `LICENSE` | cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30 |

Why vendored rather than `pip install git+…@sha` in-job: the package declares `transformers>=5.5`, which pip
would pull into the 4.57.1 pin venv that Molmo2-4B / Molmo2-ER require; the fit only needs torch + numpy.
Import path: `infra/pg/stagediff_lens.py` inserts this directory's parent (`infra/pg/vendor`) on `sys.path`.
