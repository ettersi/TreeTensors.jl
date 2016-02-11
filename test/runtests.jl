using Tensors
using TreeTensors
using Base.Test

n = ModeTree!
mtree = link!(
    n(
        n([Mode(:a,2)]), 
        n([Mode(:b,2)])
    ), 
    n(
        n([Mode(:c,2)]), 
        n([Mode(:d,2)])
    )
)
ranks = () -> [e => rand(1:5) for e in edges(mtree, root_to_leaves)]

x = rand(mode(mtree))
y = contract!(decompose(x, mtree, maxrank()))
@test_approx_eq_eps(norm(x-y), 0, 1e-14)

x = rand(mtree, ranks())
y = rand(mtree, ranks())
@test_approx_eq_eps(norm(contract(x) + contract(y) - contract!(x+y)), 0, 1e-12)

x = rand(mtree, ranks())
@test_approx_eq_eps(norm(2*x - truncate!(x+x, adaptive(1e-12))), 0, 1e-12)
