using Tensors
using TreeTensors
using Base.Test

n = ModeTree!
mtree = link!(
    n(
        n([Mode(:a,2)]), 
        n([Mode(:b,2)])
    ), 
    n([Mode(:c,2)], 
        n([Mode(:d,2)])
    )
)
randranks(mtree) = Dict(e => rand(1:5) for e in edges(mtree, root_to_leaves))
maxranks(mtree) = Dict(e => sqrt(msize(mode(mtree))) for e in edges(mtree, root_to_leaves))

x = rand(Complex{Float64},mode(mtree))
y = contract!(decompose(x, mtree, maxrank()))
@test_approx_eq_eps(norm(x-y), 0, 1e-14)

x = rand(Complex{Float64}, mtree, randranks(mtree))
y = rand(Complex{Float64}, mtree, randranks(mtree))
@test_approx_eq_eps(norm(contract(x) + contract(y) - contract!(x+y)), 0, 1e-10)

x = rand(Complex{Float64}, square(mtree), randranks(mtree))
y = rand(Complex{Float64}, square(mtree), randranks(mtree))
@test_approx_eq_eps(norm(contract(x)*contract(y) - contract!(x*y)), 0, 1e-10)

x = rand(Complex{Float64}, mtree, randranks(mtree))
@test_approx_eq_eps(norm(2*x - truncate!(x+x, adaptive(1e-12))), 0, 1e-10)

x = rand(Complex{Float64}, mtree, randranks(mtree))
y = rand(Complex{Float64}, mtree, randranks(mtree))
@test_approx_eq_eps(dot(x,y), dot(contract(x), contract(y)), 1e-10)

A = rand(Complex{Float64}, square(mtree), randranks(mtree))
b = rand(Complex{Float64}, mtree, randranks(mtree))
x = rand(Complex{Float64}, mtree, maxranks(mtree))
x,hist = als_solve!(x,A,b, DenseSolver())
@test_approx_eq_eps(norm(contract(A)*contract(x) - contract(b)), 0, 1e-10)

# GMRES doesn't work because IterativeSolvers.gmres! does 
# not do restart = min(restart, size(x))
#x = rand(Complex{Float64}, mtree, maxranks(mtree))
#x,hist = als_solve!(x,A,b, GMRES(tol=1e-15, restart = 1000))
#@test_approx_eq_eps(norm(contract(A)*contract(x) - contract(b)), 0, 1e-12)

x = rand(Complex{Float64}, mtree, randranks(mtree))
y = rand(Complex{Float64}, mtree, randranks(mtree))
z = rand(Complex{Float64}, mtree, maxranks(mtree))
z,hist = als_sum!(z,[x,y])
@test_approx_eq_eps(norm(contract(x) + contract(y) - contract(z)), 0, 1e-10)

A = rand(Complex{Float64}, square(mtree), randranks(mtree))
x = rand(Complex{Float64}, mtree, randranks(mtree))
y = rand(Complex{Float64}, mtree, randranks(mtree))
z = rand(Complex{Float64}, mtree, maxranks(mtree))
z,hist = als_axpy!(z,A,x,y)
@test_approx_eq_eps(norm(contract(A)*contract(x) + contract(y) - contract(z)), 0, 1e-10)
