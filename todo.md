# Todo

## Validation (2026-03-11)

- [x] Run full unit test suite (`julia --project=. test/runtests.jl`).
- [x] Smoke-run representative examples in `examples/1D`, `examples/2D`, and `examples/3D`.
- [ ] Fix failing `examples/2D/Diffusion/Heat_2ph.jl` manufactured setup.
- [x] Sync `examples/README.md` paths with actual file layout.

## Next technical work

- [ ] Integrate active-DOF pruning as a default/optional solve path.
- [ ] Optimize Schur-elimination implementation to reduce allocations.
- [ ] Add a CI smoke job for a short example subset.
