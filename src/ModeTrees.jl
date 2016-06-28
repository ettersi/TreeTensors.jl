export ModeTree, ModeTree!, link!, tree, mode, unsquare


# Data structure

abstract AbstractModeTree

typealias ModeDict Dict{Tree,Vector{Mode}}
immutable ModeTree <: AbstractModeTree
    tree::Tree
    free_modes::ModeDict
end
immutable SquaredModeTree <: AbstractModeTree
    mtree::ModeTree
end


# Basic functions

tree(t::ModeTree) = t.tree
tree(t::SquaredModeTree) = tree(t.mtree)

Tensors.square(t::ModeTree) = SquaredModeTree(t)
unsquare(t::AbstractModeTree) = t
unsquare(t::SquaredModeTree) = t.mtree

for f in (:getindex, :setindex!) @eval $f(x::ModeTree, args...) = $f(x.free_modes, args...) end
getindex(x::SquaredModeTree, v) = square(x.mtree[v]) 


# Tree construction

function ModeTree!(free_modes::Vector{Mode}, children::ModeTree...) 
    r = Tree!([tree(u) for u in children]...)
    D = merge!(ModeDict(r => copy(free_modes)), [u.free_modes for u in children]...)
    return ModeTree(r,D)
end
ModeTree!(children::ModeTree...) = ModeTree!(Mode[], children...)
function ModeTree!(free_modes::Vector{Mode})
    r = Tree()
    D = ModeDict(r => free_modes)
    return ModeTree(r,D)
end
ModeTree(free_modes::Vector{Mode}, children::ModeTree...) = ModeTree!(free_modes, deepcopy(children)...)
ModeTree(children::ModeTree...) = ModeTree(Mode[], children...)

function link!(v::ModeTree, u::ModeTree)
    link!(tree(v), tree(u))
    merge!(v.free_modes, u.free_modes)
    return v
end


# Other

function Tensors.mode(t::ModeTree)
    modes = Mode[]
    for D in values(t.free_modes) append!(modes, D) end
    return modes
end
