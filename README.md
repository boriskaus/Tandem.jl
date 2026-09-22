# Tandem.jl

Run [**tandem**](https://github.com/TEAR-ERC/tandem) earthquake-cycle simulations from Julia,
on Linux, macOS and Windows, with nothing to compile.

tandem is a scalable discontinuous-Galerkin code on unstructured curvilinear grids for linear
elasticity and for **SEAS** — sequences of earthquakes and aseismic slip. This package wraps
the cross-compiled binaries in [`Tandem_jll`](https://github.com/boriskaus/Tandem_jll.jl),
handles the environment they need, builds meshes through gmsh, and ships tandem's own example
problems so that a first simulation is one function call.

> [!IMPORTANT]
> **tandem is developed by the [TEAR-ERC](https://github.com/TEAR-ERC) group** — Carsten Uphoff,
> Dave May, Alice-Agnes Gabriel, Jeena Yun, Thomas Ulrich, Nico Schliwa and Casper Pranger.
> This package only packages their work; it is not affiliated with or endorsed by them.
>
> **If you use tandem, please cite:**
>
> > Uphoff, C., May, D. A., & Gabriel, A.-A. (2023). *A discontinuous Galerkin method for
> > sequences of earthquakes and aseismic slip on multiple faults using unstructured
> > curvilinear grids.* **Geophysical Journal International**, 233(1), 586–626.
> > <https://doi.org/10.1093/gji/ggac467>
>
> Source: <https://github.com/TEAR-ERC/tandem> · Documentation:
> <https://tandem.readthedocs.io> · Licence: BSD-3-Clause.

## Installation

`Tandem_jll` is not yet in the General registry (the
[Yggdrasil recipe](https://github.com/JuliaPackaging/Yggdrasil) is still in review), so it has
to be added explicitly:

```julia
using Pkg
Pkg.add(url = "https://github.com/boriskaus/Tandem_jll.jl")
Pkg.add(url = "https://github.com/boriskaus/Tandem.jl")
```

Once `Tandem_jll` is registered the first line becomes unnecessary.

## Getting started

```julia
using Tandem

# SEAS benchmark BP1, bounded to 20 time steps so it returns in a minute
r = run_example("tandem/2d/bp1_sym"; petsc = ["-ts_max_steps", "20"])

success(r)     # true
r.log          # tandem's output
```

`run_example` copies the example into a scratch directory, builds its mesh if it needs one,
and runs the right binary. To keep the output, pass a directory:

```julia
r = run_example("tandem/3d/bp5"; dir = "bp5_run", ranks = 4,
                petsc = ["-ts_max_steps", "10"], verbose = true)
```

To run your own model, point `run_tandem` (time-dependent SEAS) or `run_static` (static
elliptic solve) at a TOML parameter file:

```julia
r = run_static("cosine.toml"; dim = 2, degree = 3, output = "out/cosine")
r.l2_error     # 3.2e-5
```

## What the package provides

| Function | Purpose |
| --- | --- |
| `run_tandem(config; …)` | Run the SEAS time-integrator (`tandem`) on a TOML parameter file. |
| `run_static(config; …)` | Run the static elliptic solver (`static`) on a TOML parameter file. |
| `run_model(config; app, …)` | The common implementation behind both; pick the app explicitly. |
| `run_example(name; …)` | Prepare and run one of the bundled examples. |
| `list_examples()` / `Tandem.example(name)` | The bundled examples and how each one gets its mesh. |
| `Tandem.prepare(name; dir)` | Copy an example into a working directory and build its mesh, without running. |
| `Tandem.generate_mesh(geo; …)` | Build a `.msh` from a gmsh `.geo` file. |
| `Tandem.gmsh_executable()` | The gmsh binary used for OpenCASCADE geometries. |
| `Tandem.examples_dir()` | Path to the bundled copy of tandem's `examples/`. |
| `Tandem.executable(app, dim, degree)` | Path to one of the 12 binaries, for direct use. |

Common keyword arguments to the run functions:

| Keyword | Meaning |
| --- | --- |
| `dim`, `degree` | Which compiled configuration to use: 2D or 3D, polynomial degree 1–3. |
| `ranks` | Number of MPI ranks; `> 1` launches through `mpiexec`. |
| `petsc` | Extra PETSc options, e.g. `["-ts_max_steps", "100"]`, appended after `--petsc`. |
| `output` | Value for tandem's `--output`; VTU/PVTU files land there. |
| `dir` | Working directory, so that relative paths in the parameter file resolve. |
| `verbose` | Stream tandem's output instead of capturing it into `result.log`. |
| `check` | Throw on failure (default), or return the result for inspection. |

A run returns a `TandemResult`: `exitcode`, `log`, and — when tandem printed them —
`l2_error`, `h1_error`, `iterations` and `dofs`.

### Degrees and dimensions

tandem compiles the spatial dimension and the polynomial degree into the binary, so
`Tandem_jll` ships twelve executables: `{tandem, static}` × `{2D, 3D}` × `p1, p2, p3`.
`dim` must match the problem; `degree` is yours to choose, and higher degrees converge faster
per degree of freedom on smooth solutions.

## Bundled examples

`examples/` is a verbatim copy of tandem's own, from the commit `Tandem_jll` is built from,
so the configurations match the binaries. `list_examples()` returns all 31 with the app, the
dimension and how the mesh is obtained.

**SEAS benchmarks** (`app = :tandem` — these integrate for thousands of simulated years, so
bound them with `-ts_max_steps` unless you mean to run the whole sequence):

| Example | Description |
| --- | --- |
| `tandem/2d/bp1_sym` | SEAS BP1: 2D antiplane, planar vertical fault, rate-and-state friction, symmetric half-domain |
| `tandem/2d/bp3` | SEAS BP3: 2D antiplane, dipping fault |
| `tandem/2d/BP6/bp6_A`, `bp6_S` | SEAS BP6: fluid-injection-induced aseismic slip, at 250 m and 50 m fault resolution |
| `tandem/3d/bp5` | SEAS BP5: 3D rate-and-state fault with a velocity-strengthening border |
| `tandem/3d/tpv102` | SCEC dynamic-rupture benchmark TPV102, run quasi-dynamically |
| `tandem/2d/mms1`, `mms3`, `tandem/3d/mms5`, `3d/plane_wave` | Manufactured solutions, for verifying convergence rates |

> The 3D benchmarks **BP5** and **TPV102** are large: most of their cost is mesh partitioning
> and operator assembly, which `-ts_max_steps` does not bound, so even a two-step run takes
> well over half an hour on one core. They are meant for `ranks = 10` and upwards. The 2D
> benchmarks and the manufactured solutions run in seconds to minutes.

**Static problems** (`app = :static`): `poisson/` and `elasticity/` hold manufactured
solutions (`cosine`, `manufactured`), embedded-boundary and singular cases, and geometry-driven
setups (`circular_hole`, `spherical_hole`, `wedge`, `dip`, `beam`).

`tandem/2d/bp1` is listed but cannot be run as bundled: upstream ships neither its mesh nor a
geometry to build one from. `Tandem.runnable(e)` reports this.

## Notes

* **Meshes.** Examples whose TOML has a `[generate_mesh]` block need no mesh step — tandem
  builds the mesh itself. The rest are meshed from a gmsh `.geo`, which `Tandem.prepare` and
  `run_example` do for you. `generate_mesh` writes MSH 2.2, the format tandem parses.
  `Tandem.prepare` also creates every directory named by an output `prefix` in the parameter
  file, which tandem validates before it will start.
* **gmsh.** Most geometries are meshed through the gmsh library in-process. Six use gmsh's
  **OpenCASCADE** kernel — including the 3D benchmarks BP5 and TPV102 — which the loadable
  version lacks: gmsh_jll 4.10 and newer require HDF5_jll < 2 while `Tandem_jll` requires
  ≥ 2.2.2, leaving only 4.9.3, built without OCC. For those, a current `gmsh_jll` is resolved
  into a project of its own under the package's scratch space on first use (one download,
  needs network) and run as a subprocess. Set `ENV["TANDEM_GMSH"]` to a `gmsh` binary, or put
  one on `PATH`, to use that instead. `Tandem.needs_external_gmsh(e)` says which examples are
  affected.
* **BLAS.** The binaries link libblastrampoline, which needs a backing library in a bare
  subprocess — and two of them, since PETSc is built with 64-bit indices (ILP64) while MUMPS
  and ScaLAPACK call LP64. The package registers Julia's `libopenblas64_` alongside
  `OpenBLAS32_jll` automatically.
* **Stack size.** Large 3D models can exhaust the default 8 MiB stack. On Linux and macOS,
  raise it in the shell you start Julia from: `ulimit -s unlimited`.
* **Lua.** Scenario parameters (friction, material properties, initial conditions) are Lua
  functions in a `.lua` file next to the TOML; tandem calls them per quadrature point. Editing
  that file is how you change a model without recompiling.

## Related

* [`Tandem_jll`](https://github.com/boriskaus/Tandem_jll.jl) — the binaries
* [`test_Tandem_jll`](https://github.com/boriskaus/test_Tandem_jll) — tandem's own pytest
  regression and convergence suite run against those binaries on Linux, macOS and Windows
