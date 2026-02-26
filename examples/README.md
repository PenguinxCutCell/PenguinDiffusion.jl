# Examples

## Steady Robin Disk

`robin_disk_steady.jl` solves a manufactured steady diffusion problem inside a disk:

- `kappa * Δu = f` in the disk
- `a*u + b*kappa*dn(u) = g` on the disk interface (Robin)

Manufactured exact solution:

- `u(r) = 1 + r^2`
- `f = -4*kappa`
- `g = a*(1 + R^2) + b*kappa*(2R)`

Implementation note:

- `CartesianOperators.RobinConstraint(a, b, g, Nd)` uses `a*u + b*dn(u) = g`.
- For the physical `a*u + b*kappa*dn(u) = g`, pass `b_eff = b*kappa` to `RobinConstraint`.

Run:

```bash
julia --project=. examples/robin_disk_steady.jl
```

The script prints:

- reduced system sizes (`n_omega`, `n_gamma`)
- volume-weighted relative L2 error on active omega cells
- max absolute error on active omega cells
