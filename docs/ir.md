# The DistIR Internal Representation

```
Module = {
    # Invariant: ops are in topological order
    ops: List[Op]
    inputs: List[Value]
    outputs: List[Value]
}

Device = {
    device_id: Int
        # Unique device ID
    device_type: String
        # Device type (e.g. "gpu")
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
    device: Device
        # The device the op is assigned to (can be None if op has not been assigned)
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
    device: DeviceID
        # Which device this value lives on
    # TODO pointer to source op that generated this value?
}

Type =
    | Tensor{shape: Optional[Shape], dtype: Type}
    | Float | Int | ...

Topology = {
    devices: List[Device]
        # The list of all devices in the topology.
    bandwidths: Dict[Device, Dict[Device, Float]]
        # The bandwidth between each device.
}

```

Notes:
- All values have a device, but this can be 0 for default device.
- Tensor shapes are optional, but we might want to enforce that we know input
    shapes. We also want to retain flexibility in the definition of shape so
    that we can handle things like flexible batch dimension.


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

```Python
# Model parallel version:
def mlp(
    wA: Tensor[(F, H), 0], wB: Tensor[(H, C), 1], x: Tensor[(B, F), 0], z: Tensor[B, 1]
):
    # Suffix indicates device (e.g. a0 is on device 0)
    a0: Tensor[(B, H), 0] = MatMul(x, wA)
    a1: Tensor[(B, H), 1] = send(a0, to_device=1)
    y: Tensor[(B, C), 1] = MatMul(a1, wB)
    l: Float[1] = Loss(y, z)
    dl: Float[1] = LossGrad(y, z, 1)
    (dwB, da1): Tuple[Tensor[(H, C), 1], Tensor[(B, H), 1]] = MatMulGrad(a1, wB, dl)
    da0: Tensor[(B, H), 0] = send(da1, to_device=0) # Should this be a Tuple?
    (dwA, _): Tuple[Tensor[(F, H), 0], _] = MatMulGrad(x, wA, da0)
    wB1: Tensor[(H, C), 1] = opt(wB, dwB)
    wA1: Tensor[(F, H), 0] = opt(wA, dwA)
    return wA1, wB1, y, l
```

TODO do we really need par? Try cost model without par, see where it breaks?
- If we don't have par, IR needs to maintain invariants:
    Ordering of ops is topologically sorted (no forward dependencies)
    Ordering specifies synchronization between devices via explicit communication ops (e.g. allreduce, send).

```Python
# Pipeline parallel version (synchronous SGD, unrolled):
def mlp(
    wA: Tensor[(F, H), 0], wB: Tensor[(H, C), 1], x: Tensor[(B, F), 0], z: Tuple[Tensor[B, 1], Tensor[B, 1]], K: Int[0]
):
    # Assume K = 2 and 2 stages split between devices 0 and 1
    xs: Tuple[Tensor(B/K, F), 0] = split(x, partitions=K, dim=0)
    zs: Tuple[Tensor(B/K, 1), 1] = split(z, partitions=K, dim=0) 

    # Underscore indicates microbatch number (e.g. a0_0 is the activation on device 0 for microbatch 0)

    ################################ TIMESTEP 0 ###############################
    # Stage 0 forward pass for microbatch 0
    a0_0: Tensor[(B/K, F), 0] = MatMul(xs[0], wA)
    a1_0: Tensor[(B/K, H), 1] = send(a0_0, to_device=1)

    ################################ TIMESTEP 1 ###############################
    (
        a0_1: Tensor[(B/K, F), 0],  # TODO notation: which one is device and which one is microbatch? Use @?
        a1_1: Tensor[(B/K, H), 1],
        y_0: Tensor[(B/K, C), 1],
        l_1: Float[1]
    ) = par(
        # Stage 0 forward pass for microbatch 1
        (MatMul(xs[1], wA), 0), (send(a0_1, to_device=1), 0),
        # Stage 1 forward pass for microbatch 0
        (MatMul(a1_0, wB), 1), (Loss(y_1, zs[1]), 1) # Should loss be run in a separate timestep?
    )

    ################################ TIMESTEP 2 ###############################
    # Stage 1 forward pass for microbatch 1
    y_1: Tensor[(B/K, C), 1] = MatMul(a1_1, wB)
    l_1: Float[1] = Loss(y_1, zs[1])

    ################################ TIMESTEP 3 ###############################
    # Stage 1 backward pass for microbatch 0
    dl_0: Float[1] = LossGrad(y_0, zs[0], 1)
    (dwB_0, da1_0): Tuple[Tensor[(H, C), 1], Tensor[(B/K, H), 1]] = MatMulGrad(a1_0, wB, dl_0)
    da0_0: Tensor[(B, H), 1] = send(da1_0, to_device=0) # Should this be a Tuple?
    
    ################################ TIMESTEP 4 ###############################
    (
        (dwA_0, _): Tuple[Tensor[(F, H), 0], _],
        dl_1: Float[1],
        (dwB_1, da1_1): Tuple[Tensor[(H, C), 1], Tensor[(B/K, H), 1]],
        da0_1: Tensor[(B, H), 0],
    ) = par(
        # Stage 0 backward pass for microbatch 0
        (MatMulGrad(xs[0], wA, da0_0), 0),
        # Stage 1 backward pass for microbatch 1
        (LossGrad(y_1, zs[1], 1), 1), (MatMulGrad(a1_1, wB, dl_1), 1), (send(da1_1, to_device=0), 1)
    )

    ################################ TIMESTEP 5 ###############################
    # Stage 0 backward pass for microbatch 1
    (dwA_1, _): Tuple[Tensor[(F, H), 0], _] = MatMulGrad(xs[1], wA, da0_1)
    

    # Aggregate gradients and loss between microbatches
    dwA: Tensor[(F, H), 0] = dwA_0 + dwA_1
    dWB: Tensor[(F, H), 1] = dwB_0 + dwB_1
    y: Tensor[(B, C), 1] = concat(y_0, y_1, dim=0)
    l: Float[1] = l_0 + l_1
    
    wB1: Tensor[(H, C), 1] = opt(wB, dwB)
    wA1: Tensor[(F, H), 0] = opt(wA, dwA)
    return wA1, wB1, y, l
```

TODO add gradient accumulation so live memory isn't proportional to number of microbatches.

```Python
# Pipeline parallel version (with PipelinePartition):
def mlp(
    wA: Tensor[(F, H), 0], wB: Tensor[(H, C), 1], x: Tensor[(B, F), 0], z: Tuple[Tensor[B, 1], Tensor[B, 1]], K: Int[0]
):
    # Assume 2 stages split between devices 0 and 1
    xs: Tuple[Tensor(B/K, F), 0] = split(x, partitions=K, dim=0)
    zs: Tuple[Tensor(B/K, 1), 1] = split(z, partitions=K, dim=0) 
    
    (
        dwAs: Tuple[Tensor[(F, H), 0], ..., Tensor[(F, H), 0]]
        dwBs: Tuple[Tensor[(H, C), 1], ..., Tensor[(H, C), 1]]
        ys: Tuple[Tensor[(B/K, C), 1], ..., Tensor[(B/K, C), 1]],
        ls: Tuple[Float[1], ..., Float[1]]
    ) = PipelinePartition(
        fn=lambda(x_i: Tensor[(B/K, F), 0], z_i: Tensor[(B/K, 1), 1]): {
            # Suffix indicates device (e.g. a0_i is on device 0)
            # Underscore indicates microbatch number (e.g. a0_0 is the activation on device 0 for microbatch 0)
            a0_i: Tensor[(B, H), 0] = MatMul(x_i, wA)
            a1_i: Tensor[(B, H), 1] = send(a0_i, to_device=1)
            y_i: Tensor[(B, C), 1] = MatMul(a1_i, wB)
            l_i: Float[1] = Loss(y_i, z_i)
            dl_i: Float[1] = LossGrad(y_i, z_i, 1)
            (dwB_i, da1_i): Tuple[Tensor[(H, C), 1], Tensor[(B, H), 1]] = MatMulGrad(a1_i, wB_i, dl_i)
            da0_i: Tensor[(B, H), 0] = send(da1_i, to_device=0) # Should this be a Tuple?
            (dwA_i, _): Tuple[Tensor[(F, H), 0], _] = MatMulGrad(x_i, wA_i, da0_i)
        }
    )

    # Aggregate gradients and loss between microbatches
    dwA: Tensor[(F, H), 0] = sum(dwAs)
    dWB: Tensor[(F, H), 1] = sum(dwBs)
    y: Tensor[(B, C), 1] = concat(ys, dim=0)
    l: Float[1] = sum(ls)

    wB1: Tensor[(H, C), 1] = opt(wB, dwB)
    wA1: Tensor[(F, H), 0] = opt(wA, dwA)
    
    return wA1, wB1, y, l
```

```Python
# With activation recomputation:
def mlp(
    wA: Tensor[(F, H), 0], wB: Tensor[(H, C), 0], x: Tensor[(B, F), 0], z: Tensor[B, 0]
):
    a: Tensor[(B, H), 0] = MatMul(x, wA)
    y: Tensor[(B, C), 0] = MatMul(a, wB)
    # Release the memory used to store activation a
    free(a)
    l: Float[0] = Loss(y, z)
    dl: Float[0] = LossGrad(y, z, 1)
    # Recompute a
    a: Tensor[(B, H), 0] = MatMul(x, wA)
    (dwB, da): Tuple[Tensor[(H, C), 0], Tensor[(B, H), 0]] = MatMulGrad(a, wB, dl)
    # Release the memory used to store activation a
    free(a)
    (dwA, _): Tuple[Tensor[(F, H), 0], _] = MatMulGrad(x, wA, da)
    wB1: Tensor[(H, C), 0] = opt(wB, dwB)
    wA1: Tensor[(F, H), 0] = opt(wA, dwA)
    return wA1, wB1, y, l
```

TODO do we want explicit free?
If so, then need to add it to input module after say importing from onnx,
and need a free after last occurrence of every value.
If not, then simulator has to do a liveness analysis. This isn't hard/expensive.
But how do we deal with inplace operations? Have an attribute on those ops?
(Let's do without free for now.)

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
