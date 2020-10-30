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
    wBs: Tuple[Tensor[(F, H), 1], ..., Tensor[(F, H), N]] = broadcast(wB, devices=[1..N])
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

TODO HP and PP

Essentially, pmap is syntactic sugar. It can be unrolled: TODO
