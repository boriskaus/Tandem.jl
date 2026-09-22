# tandem is developed by the TEAR-ERC group (https://github.com/TEAR-ERC/tandem).
# See README.md for the authors and the paper to cite. The example problems exercised
# here are theirs.
#
# The 3D benchmarks and BP6 are slow enough to be awkward on a CI runner, so they are
# gated behind TANDEM_HEAVY_TESTS=true. Everything else runs by default.

using Test, Tandem

const HEAVY = get(ENV, "TANDEM_HEAVY_TESTS", "false") == "true"
const STEPS = ["-ts_max_steps", get(ENV, "TANDEM_MAX_STEPS", "5")]

@testset "Tandem.jl" begin

@testset "binaries" begin
    for (app, dim, deg) in Tandem.executables()
        @test isfile(Tandem.executable(app, dim, deg))
    end
    @test_throws ArgumentError Tandem.executable(:nope, 2, 2)
    @test_throws ArgumentError Tandem.executable(:tandem, 4, 2)
    @test_throws ArgumentError Tandem.executable(:tandem, 2, 7)
    @test isfile(Tandem.mpiexec_path())
    env = Tandem.child_env()
    @test occursin(";", env["LBT_DEFAULT_LIBS"])          # ILP64 and LP64, not one of them
    @test env["OMP_NUM_THREADS"] == "1"
end

@testset "examples catalogue" begin
    ex = Tandem.list_examples()
    @test length(ex) >= 30
    @test all(isfile(e.toml) for e in ex)
    # Every example is runnable as bundled except bp1, for which upstream ships no mesh.
    @test [e.name for e in ex if !Tandem.runnable(e)] == ["tandem/2d/bp1"]

    bp1 = Tandem.example("bp1_sym")
    @test bp1.name == "tandem/2d/bp1_sym"
    @test bp1.app === :tandem && bp1.dim == 2
    @test bp1.mesh_file == "bp1_sym.msh" && bp1.geo !== nothing
    @test Tandem.example("poisson/2d/cosine").app === :static
    @test Tandem.example("tandem/3d/bp5").dim == 3
    @test Tandem.example("mms5").generate_mesh                 # meshed by tandem itself
    @test Tandem.example("bp6_A").mesh_order == 2              # curvilinear, per upstream
    @test_throws ArgumentError Tandem.example("no-such-example")
    @test_throws ArgumentError Tandem.example("cosine")        # ambiguous across families
    @test_throws ArgumentError Tandem.options_file("nope")
    @test isfile(Tandem.options_file("lu_mumps"))
end

@testset "mesh generation" begin
    d = mktempdir()
    geo = joinpath(Tandem.examples_dir(), "tandem", "2d", "bp1_sym.geo")
    msh = Tandem.generate_mesh(geo; output = joinpath(d, "bp1_sym.msh"))
    @test isfile(msh)
    # tandem parses MSH 2.2 only; a msh4 header here means silent failure downstream.
    @test occursin("2.2", first(eachline(msh), 2)[2])
    @test_throws ArgumentError Tandem.generate_mesh(joinpath(d, "absent.geo"))

    # prepare() must leave a directory tandem can be pointed straight at
    wd, toml = Tandem.prepare("tandem/2d/bp3")
    @test isfile(toml) && isdir(joinpath(wd, "output"))
    @test isfile(joinpath(wd, "bp3.msh"))
    @test_throws ErrorException Tandem.prepare("tandem/2d/bp1")
end

# https://tandem.readthedocs.io/en/latest/getting-started/examples.html
@testset "documentation examples" begin
    @testset "elasticity cosine, p1-p3" begin
        errs = Float64[]
        for p in 1:3
            r = run_example("elasticity/2d/cosine"; degree = p)
            @test success(r)
            @test r.l2_error !== nothing
            r.l2_error === nothing || push!(errs, r.l2_error)
        end
        # High-order convergence: the error must fall as the polynomial degree rises.
        @test length(errs) == 3 && errs[1] > errs[2] > errs[3]
    end

    # A direct MUMPS factorisation instead of the iterative solve: one "iteration", and
    # the same answer. The only place anything here exercises MUMPS through PETSc.
    @testset "MUMPS direct solve" begin
        it = run_example("elasticity/2d/cosine"; degree = 2)
        lu = run_example("elasticity/2d/cosine"; degree = 2,
                         petsc = ["-options_file", Tandem.options_file("lu_mumps")])
        @test success(lu)
        @test lu.iterations == 1
        @test lu.l2_error !== nothing && it.l2_error !== nothing
        lu.l2_error === nothing || @test isapprox(lu.l2_error, it.l2_error; rtol = 1e-10)
    end

    @testset "poisson manufactured solution" begin
        r = run_example("poisson/2d/manufactured"; degree = 3)
        @test success(r)
        @test r.l2_error !== nothing && r.l2_error < 1e-3
    end

    # A mesh read from a bundled .msh rather than built by [generate_mesh] or by us,
    # i.e. the GMSH parser rather than the internal mesher.
    @testset "mesh from file" begin
        r = run_example("poisson/2d/embedded_half_msh"; degree = 2)
        @test success(r)
    end

    @testset "output files" begin
        wd, _ = Tandem.prepare("poisson/2d/cosine")
        r = run_model("cosine.toml"; app = :static, dim = 2, degree = 2,
                      dir = wd, output = "output/cosine")
        @test success(r)
        @test isfile(joinpath(wd, "output", "cosine.pvtu"))
    end
end

@testset "SEAS benchmarks (2D)" begin
    for name in ("tandem/2d/bp1_sym", "tandem/2d/bp3")
        @testset "$name" begin
            r = run_example(name; petsc = STEPS)
            @test success(r)
            @test occursin("tandem version", r.log)
            @test !occursin("Segmentation", r.log)
        end
    end

    @testset "bp1_sym on 2 MPI ranks" begin
        r = run_example("tandem/2d/bp1_sym"; ranks = 2, petsc = STEPS)
        @test success(r)
    end

    @testset "manufactured SEAS solution" begin
        r = run_example("tandem/2d/mms1"; degree = 2, petsc = STEPS)
        @test success(r)
    end
end

@testset "SEAS benchmarks (3D)" begin
    if !HEAVY
        @info "skipping 3D benchmarks; set TANDEM_HEAVY_TESTS=true to run them"
    else
        for name in ("tandem/3d/bp5", "tandem/3d/tpv102")
            @testset "$name" begin
                r = run_example(name; petsc = ["-ts_max_steps", "2"])
                @test success(r)
            end
        end
    end
end

@testset "BP6 (fluid injection, QDGreen)" begin
    if !HEAVY
        @info "skipping BP6; set TANDEM_HEAVY_TESTS=true to run it"
    else
        r = run_example("tandem/2d/BP6/bp6_A"; petsc = ["-ts_max_steps", "2"])
        @test success(r)
    end
end

@testset "error handling" begin
    wd, _ = Tandem.prepare("poisson/2d/cosine")
    @test_throws ArgumentError run_static("absent.toml"; dir = wd)
    @test_throws ArgumentError run_static("cosine.toml"; dir = wd, ranks = 0)
    bad = run_static("cosine.toml"; dir = wd, check = false, petsc = ["-nonsense_option_xyz"])
    @test bad isa TandemResult
    @test_throws ErrorException run_static("cosine.toml"; dir = wd,
                                           args = ["--not-a-flag"])
end

end
