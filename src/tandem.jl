"""
    tandem

Run [tandem](https://github.com/TEAR-ERC/tandem) simulations from Julia.

tandem is a discontinuous-Galerkin code for sequences of earthquakes and aseismic
slip (SEAS), developed by the TEAR-ERC group. This package wraps the cross-compiled
binaries in `tandem_jll`, sets up the environment they need, generates meshes with
`gmsh_jll`, and ships tandem's own example problems.

If you use tandem, cite Uphoff, May & Gabriel (2023), *Geophys. J. Int.* 233(1),
586-626, <https://doi.org/10.1093/gji/ggac467>. See the README for the full author
list and links.
"""
module tandem

using tandem_jll, gmsh_jll, OpenBLAS32_jll
using Scratch

# gmsh_jll 4.10+ requires HDF5_jll < 2 while tandem_jll requires >= 2.2.2, so the only
# version that can share an environment with the tandem binaries is 4.9.3 -- which ships
# libgmsh but no `gmsh` executable. Drive the library in process instead.
include(gmsh_jll.gmsh_api)

export run_tandem, run_static, run_model, run_example, list_examples, TandemResult

const DIMENSIONS = (2, 3)
const DEGREES = (1, 2, 3)

# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

const PATHSEP = Sys.iswindows() ? ';' : ':'
const DLEXT = Sys.iswindows() ? "dll" : Sys.isapple() ? "dylib" : "so"

"""
    backing_blas_libs() -> String

The `LBT_DEFAULT_LIBS` value handed to tandem.

The binaries link libblastrampoline, which forwards to whatever BLAS/LAPACK is
registered at run time. Inside Julia that is done for us, but tandem is a separate
process and starts with no backing library at all. Two are needed, not one: PETSc
is built with 64-bit indices and calls the ILP64 interface, while MUMPS and
ScaLAPACK call LP64. So we register Julia's own `libopenblas64_` together with
`OpenBLAS32_jll`.
"""
function backing_blas_libs()
    ilp64 = Sys.iswindows() ? joinpath(Sys.BINDIR, "libopenblas64_.dll") :
            joinpath(Sys.BINDIR, "..", "lib", "julia", "libopenblas64_.$DLEXT")
    return join((ilp64, OpenBLAS32_jll.libopenblas_path), ";")
end

"The MPI JLL `tandem_jll` was built against on this platform."
function mpi_module()
    for sym in (:MPICH_jll, :OpenMPI_jll, :MPItrampoline_jll, :MicrosoftMPI_jll)
        isdefined(tandem_jll, sym) && return getproperty(tandem_jll, sym)
    end
    error("tandem_jll exposes no MPI implementation on this platform")
end

"""
    mpiexec_path() -> String

Absolute path to the MPI launcher matching the tandem binaries. Note that `mpirun`
does not exist on Windows (MS-MPI ships only `mpiexec`), and that OpenMPI-only flags
such as `--oversubscribe` are rejected by MPICH's Hydra.
"""
mpiexec_path() = mpi_module().mpiexec().exec[1]

"""
    child_env(; threads = 1, extra = Dict{String,String}()) -> Dict{String,String}

Environment for a tandem subprocess: the JLL library paths *prepended to* (never
replacing) the inherited ones, the MPI launcher on `PATH`, a BLAS backing library
for libblastrampoline, and an OpenMP thread count.
"""
function child_env(; threads::Integer = 1, extra = Dict{String,String}())
    libkey = tandem_jll.JLLWrappers.LIBPATH_env
    libdirs = unique(vcat(tandem_jll.LIBPATH_list..., mpi_module().LIBPATH_list...,
                          split(get(ENV, libkey, ""), PATHSEP)))
    bindirs = (dirname(mpiexec_path()), get(ENV, "PATH", ""))
    env = Dict{String,String}(
        "PATH" => join(filter(!isempty, collect(bindirs)), PATHSEP),
        libkey => join(filter(!isempty, libdirs), PATHSEP),
        "LBT_DEFAULT_LIBS" => backing_blas_libs(),
        "OMP_NUM_THREADS" => string(threads),
    )
    merge!(env, Dict{String,String}(string(k) => string(v) for (k, v) in pairs(extra)))
    return env
end

# ---------------------------------------------------------------------------
# executables
# ---------------------------------------------------------------------------

"""
    executable(app, dim, degree) -> String

Path to one of the 12 binaries in `tandem_jll`. `app` is `:tandem` (the SEAS
time-integrator) or `:static` (the static solver), `dim` is 2 or 3 and `degree` is
the polynomial degree 1, 2 or 3. Both are compiled in, which is why there is a
binary per combination rather than a run-time flag.
"""
function executable(app::Symbol, dim::Integer, degree::Integer)
    app in (:tandem, :static) || throw(ArgumentError("app must be :tandem or :static, got $app"))
    dim in DIMENSIONS || throw(ArgumentError("dim must be 2 or 3, got $dim"))
    degree in DEGREES || throw(ArgumentError("degree must be 1, 2 or 3, got $degree"))
    return getproperty(tandem_jll, Symbol(app, "_", dim, "d_p", degree))().exec[1]
end

"All 12 (app, dim, degree) combinations `tandem_jll` provides."
executables() = [(app, d, p) for app in (:tandem, :static) for d in DIMENSIONS for p in DEGREES]

# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

"""
    TandemResult

Outcome of a tandem run. `log` holds the combined stdout/stderr (empty when the run
was made with `verbose = true`, which streams instead); the remaining fields are
parsed out of it and are `nothing` when the run did not print them.

Use `success(result)` to check whether the run completed.
"""
struct TandemResult
    app::Symbol
    dim::Int
    degree::Int
    ranks::Int
    config::String
    dir::String
    outdir::String
    exitcode::Int
    log::String
    l2_error::Union{Float64,Nothing}
    h1_error::Union{Float64,Nothing}
    iterations::Union{Int,Nothing}
    dofs::Union{Int,Nothing}
end

Base.success(r::TandemResult) = r.exitcode == 0

function Base.show(io::IO, r::TandemResult)
    print(io, "TandemResult(", r.app, " ", r.dim, "D p", r.degree,
          ", ranks=", r.ranks, ", ", success(r) ? "ok" : "exit $(r.exitcode)")
    r.l2_error === nothing || print(io, ", L2=", r.l2_error)
    r.iterations === nothing || print(io, ", its=", r.iterations)
    print(io, ")")
end

_float(log, pat) = (m = match(pat, log); m === nothing ? nothing : parse(Float64, m[1]))
_int(log, pat) = (m = match(pat, log); m === nothing ? nothing : parse(Int, m[1]))

# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

"""
    run_model(config; app, dim = 2, degree = 2, ranks = 1, output = nothing,
              dir = dirname(config), petsc = String[], args = String[],
              threads = 1, env = Dict(), verbose = false, check = true)

Run a tandem model described by the TOML parameter file `config` and return a
[`TandemResult`](@ref).

* `app` -- `:tandem` for the SEAS time-integrator, `:static` for the static solver.
* `dim`, `degree` -- select the compiled-in configuration; must match the problem.
* `ranks` -- MPI ranks. `ranks > 1` launches through `mpiexec`.
* `output` -- value for tandem's `--output`; VTU/PVTU files land there.
* `dir` -- working directory. Defaults to the parameter file's own directory, so
  that relative paths inside it (meshes, Lua scripts) resolve.
* `petsc` -- extra PETSc options, e.g. `["-ts_max_steps", "20"]`. They are passed
  after `--petsc`, which must come last on tandem's command line.
* `args` -- extra tandem arguments, placed before `--petsc`.
* `verbose` -- stream output to the terminal instead of capturing it.
* `check` -- throw if the run fails; set `false` to inspect `result.log` yourself.

The SEAS benchmarks integrate for thousands of years. Bound them with
`petsc = ["-ts_max_steps", "100"]` unless you mean to run the whole sequence.
"""
function run_model(config::AbstractString;
                   app::Symbol = :tandem, dim::Integer = 2, degree::Integer = 2,
                   ranks::Integer = 1, output = nothing,
                   dir::AbstractString = dirname(abspath(config)),
                   petsc::AbstractVector{<:AbstractString} = String[],
                   args::AbstractVector{<:AbstractString} = String[],
                   threads::Integer = 1, env = Dict{String,String}(),
                   verbose::Bool = false, check::Bool = true)
    ranks >= 1 || throw(ArgumentError("ranks must be >= 1"))
    isdir(dir) || throw(ArgumentError("working directory does not exist: $dir"))
    isfile(joinpath(dir, config)) ||
        throw(ArgumentError("parameter file not found: $(joinpath(dir, config))"))

    argv = String[executable(app, dim, degree), String(config)]
    outdir = output === nothing ? "" : abspath(joinpath(dir, String(output)))
    if !isempty(outdir)
        mkpath(dirname(outdir))
        append!(argv, ["--output", outdir])
    end
    append!(argv, args)
    isempty(petsc) || (push!(argv, "--petsc"); append!(argv, petsc))
    ranks > 1 && (argv = vcat([mpiexec_path(), "-n", string(ranks)], argv))

    buf = IOBuffer()
    cmd = addenv(Cmd(Cmd(argv); dir = dir), child_env(; threads, extra = env))
    proc = run(pipeline(ignorestatus(cmd);
                        stdout = verbose ? stdout : buf,
                        stderr = verbose ? stderr : buf))
    log = verbose ? "" : String(take!(buf))

    r = TandemResult(app, Int(dim), Int(degree), Int(ranks), String(config), abspath(dir),
                     outdir, proc.exitcode, log,
                     _float(log, r"L2 error:\s*([0-9.eE+-]+)"),
                     _float(log, r"H1 error:\s*([0-9.eE+-]+)"),
                     _int(log, r"Iterations:\s*([0-9]+)"),
                     _int(log, r"DOFs:\s*([0-9]+)"))
    if check && !success(r)
        error("tandem exited with code $(r.exitcode)\n" *
              (isempty(log) ? "(output was streamed; rerun with verbose = false to capture it)" : log))
    end
    return r
end

"""
    run_tandem(config; kwargs...)

[`run_model`](@ref) with `app = :tandem`: the SEAS time-integrator, used by the BP
benchmarks and the TPV rupture scenarios.
"""
run_tandem(config; kwargs...) = run_model(config; app = :tandem, kwargs...)

"""
    run_static(config; kwargs...)

[`run_model`](@ref) with `app = :static`: the static elliptic solver, used by the
Poisson and elasticity examples and by the manufactured-solution convergence tests.
"""
run_static(config; kwargs...) = run_model(config; app = :static, kwargs...)

# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------

"""
    external_gmsh() -> Union{String,Nothing}

Path to a `gmsh` executable the user supplied, from `ENV["TANDEM_GMSH"]` or from
`PATH`. Takes precedence over both the bundled library and the private install that
[`gmsh_executable`](@ref) provisions.
"""
function external_gmsh()
    e = get(ENV, "TANDEM_GMSH", "")
    isempty(e) && return Sys.which("gmsh")
    isfile(e) || throw(ArgumentError("TANDEM_GMSH is set to $e, which is not a file"))
    return e
end

"Whether a .geo needs the OpenCASCADE kernel, which the bundled gmsh lacks."
needs_occ(geo::AbstractString) = occursin("OpenCASCADE", read(geo, String))

const GMSH_EXE = Ref{Union{String,Nothing}}(nothing)
const GMSH_LIBPATH = Ref{Vector{String}}(String[])

"""
    gmsh_executable(; install = true) -> Union{String,Nothing}

Path to a gmsh binary that supports the OpenCASCADE kernel, or `nothing` if none can
be obtained.

`gmsh_jll` 4.10 and newer require `HDF5_jll < 2` while `tandem_jll` requires 2.2.2 or
newer, so the version this package can load directly is 4.9.3 -- which is built
without OCC. Six of the bundled geometries need it, including the 3D SEAS benchmarks
BP5 and TPV102.

Rather than make those unreachable, a current `gmsh_jll` is resolved into a project
of its own under this package's scratch space, where the HDF5 conflict cannot arise,
and its executable is used as a subprocess. That happens once, on first use, and
needs network access; set `install = false` to check availability without it, or set
`ENV["TANDEM_GMSH"]` to skip it entirely.
"""
function gmsh_executable(; install::Bool = true)
    e = external_gmsh()
    e === nothing || return e
    GMSH_EXE[] === nothing || return GMSH_EXE[]

    dir = @get_scratch!("gmsh")
    stamp = joinpath(dir, "resolved.txt")
    if !isfile(stamp)
        install || return nothing
        script = joinpath(dir, "resolve.jl")
        write(script, """
            using Pkg
            Pkg.add("gmsh_jll"; io = devnull)
            using gmsh_jll
            open(ARGS[1], "w") do io
                println(io, gmsh_jll.gmsh_path)
                foreach(p -> println(io, p), gmsh_jll.LIBPATH_list)
            end
            """)
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$dir $script $stamp`
        buf = IOBuffer()
        pr = run(pipeline(ignorestatus(cmd); stdout = buf, stderr = buf))
        if pr.exitcode != 0 || !isfile(stamp)
            # Silence here once cost a CI run: the OpenCASCADE geometries were skipped
            # with no indication of why. Say what happened, then carry on without gmsh.
            @warn """Could not provision a gmsh with OpenCASCADE support; geometries \
                     that need it cannot be meshed. Set ENV["TANDEM_GMSH"] to a gmsh \
                     binary to bypass this.""" exitcode=pr.exitcode dir output=String(take!(buf))
            return nothing
        end
    end
    lines = filter(!isempty, strip.(readlines(stamp)))
    isempty(lines) && return nothing
    isfile(first(lines)) || return nothing
    GMSH_LIBPATH[] = String.(lines[2:end])
    return GMSH_EXE[] = String(first(lines))
end

"""
    generate_mesh(geo; output = nothing, dim = 2, order = 1, format = "msh2",
                  options = Dict(), verbose = false)

Build a `.msh` mesh from a gmsh `.geo` file and return its path.

gmsh is driven through its library API in this process, so no `gmsh` executable is
normally needed. Geometries built on the **OpenCASCADE** kernel are the exception and
go through [`gmsh_executable`](@ref); a `gmsh` in `ENV["TANDEM_GMSH"]` or on `PATH`
is used in preference to either.

tandem reads MSH 2.2, so `format` defaults to `"msh2"`; newer formats are not
parsed. `order = 2` asks gmsh for curvilinear elements, which tandem supports and
some benchmarks rely on. `options` entries become gmsh `-setnumber` arguments, which
is how the geometries parameterise resolution, e.g. `options = Dict("hf" => 0.25)`.

Examples with a `[generate_mesh]` block in their TOML need no mesh step -- tandem
builds those internally.
"""
function generate_mesh(geo::AbstractString; output = nothing, dim::Integer = 2,
                       order::Integer = 1, format::AbstractString = "msh2",
                       options = Dict(), verbose::Bool = false)
    isfile(geo) || throw(ArgumentError("geometry file not found: $geo"))
    msh = output === nothing ? string(first(splitext(abspath(geo))), ".msh") :
          abspath(String(output))
    mkpath(dirname(msh))

    ext = needs_occ(geo) ? gmsh_executable() : external_gmsh()
    if ext !== nothing
        argv = String[ext, "-$(dim)", abspath(geo), "-o", msh, "-format", format,
                      "-order", string(order)]
        for (k, v) in pairs(options)
            append!(argv, ["-setnumber", string(k), string(v)])
        end
        buf = IOBuffer()
        libkey = tandem_jll.JLLWrappers.LIBPATH_env
        env = Dict(libkey => join(filter(!isempty,
                                  vcat(GMSH_LIBPATH[], get(ENV, libkey, ""))), PATHSEP))
        cmd = addenv(Cmd(Cmd(argv); dir = dirname(abspath(geo))), env)
        pr = run(pipeline(ignorestatus(cmd); stdout = verbose ? stdout : buf,
                          stderr = verbose ? stderr : buf))
        pr.exitcode == 0 || error("gmsh failed with code $(pr.exitcode)\n" * String(take!(buf)))
        isfile(msh) || error("gmsh reported success but produced no mesh at $msh")
        return msh
    end

    needs_occ(geo) && error("""
        $(basename(geo)) needs gmsh's OpenCASCADE kernel, which the gmsh library this
        package can load (4.9.3, pinned by tandem_jll's HDF5 requirement) lacks, and a
        current gmsh could not be provisioned -- see `tandem.gmsh_executable`.

        Install gmsh and put it on PATH or in ENV["TANDEM_GMSH"], or retry with network
        access. Examples whose TOML has a [generate_mesh] block need no gmsh at all.""")

    # `-setnumber` has no API call in gmsh 4.9; passing it through initialize's argv is
    # the supported equivalent, and is what sets a geometry's DefineConstant values.
    argv = String["gmsh"]
    for (k, v) in pairs(options)
        append!(argv, ["-setnumber", string(k), string(v)])
    end

    gmsh.initialize(argv)
    try
        gmsh.option.setNumber("General.Verbosity", verbose ? 5 : 2)
        gmsh.open(abspath(geo))
        gmsh.model.mesh.generate(dim)
        order == 1 || gmsh.model.mesh.setOrder(order)
        format == "msh2" && gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(msh)
    finally
        gmsh.finalize()
    end
    isfile(msh) || error("gmsh reported success but produced no mesh at $msh")
    return msh
end

# ---------------------------------------------------------------------------
# examples
# ---------------------------------------------------------------------------

"""
    examples_dir() -> String

The bundled copy of tandem's `examples/` tree, taken verbatim from the commit
`tandem_jll` is built from. See `PROVENANCE.md` there.
"""
examples_dir() = normpath(joinpath(@__DIR__, "..", "examples"))

"""
    options_file(name) -> String

Path to one of the PETSc options files tandem bundles under `examples/options`, for
use as `petsc = ["-options_file", options_file("lu_mumps")]`.

`lu_mumps` swaps the default iterative solve for a direct MUMPS factorisation;
`mg_cheby` is a multigrid preconditioner; `rk45` selects an explicit time integrator;
the `eigdeflate_*` files configure the eigenvalue-deflation preconditioner.
"""
function options_file(name::AbstractString)
    f = joinpath(examples_dir(), "options", endswith(name, ".cfg") ? name : name * ".cfg")
    isfile(f) || throw(ArgumentError("no options file \"$name\"; available: " *
        join(first.(splitext.(readdir(joinpath(examples_dir(), "options")))), ", ")))
    return f
end

"""
    Example

A bundled example. Besides its `name` (e.g. `"tandem/2d/bp1_sym"`), the `app` and
`dim` it runs with and its parameter file `toml`, this records how the example gets
its mesh, which is the only fiddly part of running one:

* `generate_mesh = true` -- the TOML has a `[generate_mesh]` block and tandem builds
  the mesh itself. Nothing to do.
* `mesh` is a bundled `.msh` -- ready to use.
* `geo` is set -- the mesh is built from that gmsh geometry, with `mesh_options`.
* none of the above -- `runnable` is `false`; upstream ships neither mesh nor
  geometry for this example.
"""
struct Example
    name::String
    app::Symbol
    dim::Int
    toml::String
    mesh_file::Union{String,Nothing}
    mesh::Union{String,Nothing}
    geo::Union{String,Nothing}
    mesh_options::Dict{String,Any}
    mesh_order::Int
    generate_mesh::Bool
    description::String
end

"Whether this example can be run as bundled -- see [`Example`](@ref)."
runnable(e::Example) = e.generate_mesh || e.mesh !== nothing || e.geo !== nothing

"""
    needs_external_gmsh(e) -> Bool

Whether this example's geometry uses gmsh's OpenCASCADE kernel, which the gmsh
library this package loads directly lacks. Meshing it goes through
[`gmsh_executable`](@ref), which provisions a current gmsh on first use.
"""
needs_external_gmsh(e::Example) = e.geo !== nothing && needs_occ(e.geo)

function Base.show(io::IO, e::Example)
    print(io, "Example(\"", e.name, "\", ", e.app, ", ", e.dim, "D, ",
          e.generate_mesh ? "[generate_mesh]" :
          e.mesh !== nothing ? "mesh bundled" :
          e.geo !== nothing ? "mesh from " * basename(e.geo) : "NO MESH", ")")
end

const DESCRIPTIONS = Dict(
    "tandem/2d/bp1_sym"        => "SEAS benchmark BP1: 2D antiplane, planar vertical fault, rate-and-state friction, symmetric half-domain",
    "tandem/2d/bp1"            => "SEAS benchmark BP1 on the full domain (upstream ships no .geo for it, so no mesh can be built)",
    "tandem/2d/bp3"            => "SEAS benchmark BP3: 2D antiplane, dipping fault",
    "tandem/2d/tutorial"       => "Geometry for the 'my first model' tutorial (tutorial.lua is broken in v1.2.0)",
    "tandem/3d/bp5"            => "SEAS benchmark BP5: 3D, rate-and-state fault with a velocity-strengthening border",
    "tandem/3d/tpv102"         => "SCEC dynamic-rupture benchmark TPV102, run quasi-dynamically",
    "poisson/2d/mms1"          => "2D Poisson manufactured solution, for convergence testing",
    "poisson/3d/mms5"          => "3D Poisson manufactured solution",
    "elasticity/2d/cosine"     => "2D elasticity with a cosine manufactured solution (the documentation's worked example)",
    "elasticity/2d/mms3"       => "2D elasticity manufactured solution",
    "elasticity/3d/plane_wave" => "3D elastic plane wave",
)

"""
    MESH_OPTIONS

gmsh settings for the few geometries that need more than the defaults, keyed by
example name. BP6 takes its fault resolution from the geometry's `hf` constant and
needs second-order elements, exactly as upstream's `generate_mesh.sh` does.
"""
const MESH_OPTIONS = Dict{String,Tuple{Dict{String,Any},Int}}(
    "tandem/2d/BP6/bp6_A" => (Dict{String,Any}("hf" => 0.250), 2),
    "tandem/2d/BP6/bp6_S" => (Dict{String,Any}("hf" => 0.050), 2),
)

# tandem's TOML is small and regular enough to read with a regex; pulling in a TOML
# parser only to find `mesh_file` would not earn its dependency.
function _mesh_file(txt::AbstractString)
    m = match(r"(?m)^\s*mesh_file\s*=\s*[\"']([^\"']+)[\"']", txt)
    return m === nothing ? nothing : String(m[1])
end

"""
    list_examples() -> Vector{Example}

Every bundled example that has a TOML parameter file.
"""
function list_examples()
    root = examples_dir()
    out = Example[]
    for (dir, _, files) in walkdir(root), f in sort(files)
        endswith(f, ".toml") || continue
        toml = joinpath(dir, f)
        rel = replace(relpath(dir, root), '\\' => '/')
        stem = first(splitext(f))
        name = "$rel/$stem"
        txt = read(toml, String)
        # The SEAS app is the one that integrates in time; everything else is static.
        app = (startswith(rel, "tandem") || occursin("final_time", txt)) ? :tandem : :static
        dim = occursin(r"(^|/)3d(/|$)", rel) ? 3 : 2
        gen = occursin("[generate_mesh]", txt)
        mf = _mesh_file(txt)

        mesh = (mf !== nothing && isfile(joinpath(dir, mf))) ? joinpath(dir, mf) : nothing
        # Find a geometry that could produce the mesh: one named after it, one named
        # after the parameter file, or the single .geo sitting in the example's folder.
        geo = nothing
        if !gen && mesh === nothing
            geos = filter(x -> endswith(x, ".geo"), readdir(dir))
            for cand in filter(!isnothing, (mf === nothing ? nothing : first(splitext(mf)) * ".geo",
                                            stem * ".geo"))
                cand in geos && (geo = joinpath(dir, cand); break)
            end
            geo === nothing && length(geos) == 1 && (geo = joinpath(dir, only(geos)))
        end
        opts, order = get(MESH_OPTIONS, name, (Dict{String,Any}(), 1))
        push!(out, Example(name, app, dim, toml, mf, mesh, geo, opts, order, gen,
                           get(DESCRIPTIONS, name, "")))
    end
    return out
end

"""
    example(name) -> Example

Look a bundled example up by name, e.g. `example("tandem/2d/bp1_sym")`. A unique
suffix is accepted, so `example("bp1_sym")` works too.
"""
function example(name::AbstractString)
    all_ex = list_examples()
    hits = filter(e -> e.name == name || endswith(e.name, "/" * name), all_ex)
    length(hits) == 1 && return only(hits)
    isempty(hits) && throw(ArgumentError(
        "no example matching \"$name\". Available:\n  " * join(getfield.(all_ex, :name), "\n  ")))
    throw(ArgumentError("\"$name\" is ambiguous: " * join(getfield.(hits, :name), ", ")))
end
example(e::Example) = e

"""
    prepare(name; dir = mktempdir()) -> (dir, toml)

Copy an example into a writable working directory, build its mesh if it needs one,
and return the directory together with the parameter file inside it.

The bundled tree lives inside the package and is read-only, while tandem writes
output next to the parameter file, so a working copy is the normal starting point.
`prepare` also creates the `output/` subdirectory that several examples expect to
exist already.
"""
function prepare(name; dir::AbstractString = mktempdir())
    e = example(name)
    runnable(e) || error("example \"$(e.name)\" cannot be run as bundled: it needs " *
                         "$(e.mesh_file), for which upstream ships no geometry")
    mkpath(dir)
    src = dirname(e.toml)
    for f in readdir(src)
        isfile(joinpath(src, f)) && cp(joinpath(src, f), joinpath(dir, f); force = true)
    end
    # tandem validates output prefixes against the filesystem and refuses to start if
    # the directory is missing, so create one per `prefix = "..."` in the parameter file.
    mkpath(joinpath(dir, "output"))
    for m in eachmatch(r"(?m)^\s*prefix\s*=\s*[\"']([^\"']+)[\"']", read(e.toml, String))
        d = dirname(m[1])
        isempty(d) || mkpath(joinpath(dir, d))
    end
    if e.geo !== nothing
        generate_mesh(joinpath(dir, basename(e.geo));
                      output = joinpath(dir, something(e.mesh_file, "mesh.msh")),
                      dim = e.dim, order = e.mesh_order, options = e.mesh_options)
    end
    return dir, joinpath(dir, basename(e.toml))
end

"""
    run_example(name; degree = 2, dir = mktempdir(), kwargs...) -> TandemResult

[`prepare`](@ref) an example and run it. `app` and `dim` come from the example;
everything else is forwarded to [`run_model`](@ref).

```julia
using tandem
r = run_example("tandem/2d/bp1_sym"; petsc = ["-ts_max_steps", "20"])
```
"""
function run_example(name; degree::Integer = 2, dir::AbstractString = mktempdir(), kwargs...)
    e = example(name)
    wd, toml = prepare(e; dir)
    return run_model(basename(toml); app = e.app, dim = e.dim, degree, dir = wd, kwargs...)
end

end # module
