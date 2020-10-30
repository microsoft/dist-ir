# The DistIR Internal Representation

```
Module = {
    # Invariant: ops are in topological order
    ops: List[Op]
    inputs: List[Value]
    outputs: List[Value]
}

Op = {
    name: String
        # Do we need this to be unique in a module? Across all modules?
    op_type: OpType
        # The type of operator
    in_edges: List[Value]
        # Pointer to a Value object either in Module.inputs or another Op.out_edges
    out_edges: List[Value]
        # To support ops that have more than one output
    attributes: Dict[String, Any]
        # Constant data for ops, e.g. stride of convolution or devices to scatter
    submodules: List[Module]
}

OpType = 
    | MatMul | Add | Relu | ...
    | MatMulGrad | ReluGrad | ...
    | N11Loss | SGDOptimizer | ...
    | Pmap | AllReduce | ...

Value = {
    name: String
        # Again, does it need to be unique in a module?
    type: Type
    device: Int
        # Which device this value lives on
}

Type =
    | Tensor{shape: Shape, dtype: Type}
    | Float | Int | ...
```

TODO should device be a field of Value, or should each Type have a device field?

## Representing distributed computation

### Option: pmap + device annotations

Motivation: the trouble with having only pmap is that communication of data is
implicit (data is transfered to devices at the start of pmap and back at the
end). This makes it hard to represent the allreduce inside a pmap, needed for
data parallelism, especially when it comes to modelling cost or the reference
sequential executor. Ideally, the allreduce would be outside the pmap, and
it would explicitly take as arguments all the tensors (across devices) that it
operates on. This means we need a way to represent tensors that live on
different devices outside pmap bodies.

A pmap contains:
- device_var: binds a device variable d, to be used to specify types inside fn
- fn: a (sub)graph with inputs, outputs, and a body
- data: an iterable or tuple of iterables to map over

The device types of pmap's return values are determined by those of data
but their shapes must match those of the return values of fn.

```Python
# Input sequential computation:
def mlp(
    wA: Tensor[(F, H), 0], wB: Tensor[(H, C), 0], x: Tensor[(B, F), 0], z: Tensor[B, 0]
):
    a: Tensor[(B, H), 0] = MatMul(x, wA)
    y: Tensor[(B, C), 0] = MatMul(a, wB)
    l: Float[0] = Loss(y, z)
    dl: Float[0] = LossGrad(y, z, 1)
    (dwB, da): Tuple[Tensor[(H, C), 0], Tensor[(B, H), 0]] = MatMulGrad(a, wB, dl)
    (dwA, _): Tuple[Tensor[(F, H), 0], _] = MatMulGrad(x, wA, da)
    wB1: Tensor[(H, C), 0] = opt(wB, dwB)
    wA1: Tensor[(F, H), 0] = opt(wA, dwA)
    return wA1, wB1, y, l
    # TODO import this from ONNX and check if correct
```

```Python
# Data parallel version:
def mlp(
    wA: Tensor[(F, H), 0], wB: Tensor[(H, C), 0], x: Tensor[(B, F), 0], z: Tensor[B, 0]
):
    # These are finite tuples because N is compile-time const:
    xs: Tuple[Tensor[(B/N, F), 1], ..., Tensor[(B/N, F), N]] = scatter(x, dim=0, devices=[1..N])
    zs: Tuple[Tensor[(B/N), 1], ..., Tensor[(B/N), N]] = scatter(z, dim=0, devices=[1..N])
    wAs: Tuple[Tensor[(F, H), 1], ..., Tensor[(F, H), N]] = broadcast(wA, devices=[1..N])
    wBs: Tuple[Tensor[(H, C), 1], ..., Tensor[(H, C), N]] = broadcast(wB, devices=[1..N])
    # If we don't want to represent training data sharding, we can start from here:
    (
        dwAis: Tuple[Tensor[(F, H), 1], ..., Tensor[(F, H), N]],
        dwBis: Tuple[Tensor[(H, C), 1], ..., Tensor[(H, C), N]],
        ys: Tuple[Tensor[(B/N, C), 1], ..., Tensor[(B/N, C), N]],
        ls: Tuple[Float[1], ..., Float[N]],
    ) = pmap(
        device_var=d, # Need a binder here to express variable device types below
        fn=lambda (xi: Tensor[(B/N, F), d]), (zi: Tensor[(B/N), d]),
                (wAi: Tensor[(F, H), d]), (wBi: Tensor[(H, C), d]): {
            ai: Tensor[(B/N, H), d] = MatMul(xi, wAi)
            yi: Tensor[(B/N, C), d] = MatMul(ai, wBi)
            li: Float[d] = Loss(yi, zi)
            dli: Float[d] = LossGrad(yi, zi, 1)
            # The fact that N does not appear in these shapes hints that allreduce needed:
            (dwBi, dai): Tuple[Tensor[(H, C), d], Tensor[(B/N, H), d]] = MatMulGrad(ai, wBi, dli)
            (dwAi, _): Tuple[Tensor[(F, H), d], _] = MatMulGrad(xi, wAi, dai)
            return dwAi, dwBi, yi, li
        },
        (xs, zs, wAs, wBs)
    )
    dwBs: Tuple[Tensor[(H, C), 1], ..., Tensor[(H, C), N]] = allreduce(dwBis)
    dwAs: Tuple[Tensor[(F, H), 1], ..., Tensor[(F, H), N]] = allreduce(dwAis)
    (
        wA1s: Tuple[Tensor[(F, H), 1], ..., Tensor[(F, H), N]],
        wB1s: Tuple[Tensor[(H, C), 1], ..., Tensor[(H, C), N]],
    ) = pmap(
        device_var=d,
        body=lambda (wAi: Tensor[(F, H), d]), (wBi: Tensor[(H, C), d]),
                (dwAi: Tensor[(F, H), d]), (dwBi: Tensor[(H, C), d]): {
            # Note: dwAi and dwBi are now the combined gradients, replicated
            wB1i: Tensor[(H, C), d] = opt(wBi, dwBi)
            wA1i: Tensor[(F, H), d] = opt(wAi, dwAi)
        },
        (wAs, wBs, dwAs, dwBs)
    )
    # We can also skip the rest if not needed:
    wA1: Tensor[] = send(wA1s[0], to_device=0)
    wB1: Tensor[] = send(wB1s[0], to_device=0)
    y = gather(ys, ??)
    l = reduce(ls, ??)
    return wA1, wB1, y, l
    # NOTE disadvantage: needs recompilation to redeploy on different #devices
```

```Python
# Horizontal parallel version:
def mlp(
    wA: Tensor[(F, H), 0], wB: Tensor[(H, C), 0], x: Tensor[(B, F), 0], z: Tensor[B, 0]
):
    xs: Tuple[Tensor[(B, F), 1], ..., Tensor[(B, F), N]] = broadcast(x, devices=[1..N])
    zs: Tuple[Tensor[(B), 1], ..., Tensor[(B), N]] = broadcast(z, devices=[1..N])
    wAs: Tuple[Tensor[(F, H/N), 1], ..., Tensor[(F, H/N), N]] = scatter(wA, dim=1, devices=[1..N])
    wBs: Tuple[Tensor[(H/N, C), 1], ..., Tensor[(H/N, C), N]] = scatter(wB, dim=0, devices=[1..N])
    # Note that wA and wB are split on different dimensions
    (
        yis: Tuple[Tensor[(B, C), 1], ..., Tensor[(B, C), N]],
        ais: Tuple[Tensor[(B, H/N), 1], ..., Tensor[(B, H/N), N]],
    ) = pmap(
        device_var=d,
        fn=lambda (xi: Tensor[(B, F), d]), (wAi: Tensor[(F, H/N), d]),
                (wBi: Tensor[(H/N, C), d]): {
            ai: Tensor[(B, H/N), d] = MatMul(xi, wAi)
            yi: Tensor[(B, C), d] = MatMul(ai, wBi)
            return yi
        },
        (xs, wAs, wBs)
    )
    ys: Tuple[Tensor[(B, C), 1], ..., Tensor[(B, C), N]] = allreduce(yis)
    (
        wA1s: Tuple[Tensor[(F, H/N), 1], ..., Tensor[(F, H/N), N]],
        wB1s: Tuple[Tensor[(H/N, C), 1], ..., Tensor[(H/N, C), N]],
        ls: Tuple[Float[1], ..., Float[N]],
    ) = pmap(
        device_var=d,
        fn=lambda (y: Tensor[(B, C), d]), (z: Tensor[(B), d]), (ai: Tensor[(B, H/N), d])
                (wAi: Tensor[(F, H/N), d]), (wBi: Tensor[(H/N, C), d]): {
            l: Float[d] = Loss(y, z)
            dl: Float[d] = LossGrad(y, z, 1)
            (dwBi, dai): Tuple[Tensor[(H/N, C), d], Tensor[(B, H/N), d]] = MatMulGrad(ai, wBi, dl)
            (dwAi, _): Tuple[Tensor[(F, H/N), d], _] = MatMulGrad(xi, wAi, dai)
            wB1i: Tensor[(H/N, C), d] = opt(wBi, dwBi)
            wA1i: Tensor[(F, H/N), d] = opt(wAi, dwAi)
            return wA1i, wB1i, l
        },
        (ys, zs, ais, wAs, wBs)
    )
    wA1: Tensor[(F, H), 0] = gather(wA1s[0], to_device=0)
    wB1: Tensor[(H, C), 0] = gather(wB1s[0], to_device=0)
    y = send(ys[0], to_device=0)
    l = send(ls[0], to_device=0)
    return wA1, wB1, y, l
```

TODO: pipeline parallel
- tricky thing is to make the stages that execute in parallel explicit, so that cost model can be accurate
- this means no using wait/record sync events, because some intelligence is needed to infer pipeline shape from that
- brute force option is to unroll entire pipeline, but it will be nice if the stages aren't needlessly replicated
- should we have some notion of functions? so the stages can just be function calls?

#### Lowering pmap:

Essentially, pmap is syntactic sugar. It can be unrolled: TODO

## Miscellaneous:

TODO is there a more general horizontal transform, beyond pattern matching on
two layer FF or attention layers?