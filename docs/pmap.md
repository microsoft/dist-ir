# pmap

TODO: describe pmap syntax and semantics.

# Nested pmap

Input sequential DistIR function:

```python
def matmul(wA: Tensor[(F, H), 0], x: Tensor[(B, F), 0]):
    xs: Tuple[Tensor[(B/N, F), 1], Tensor[(B/N, F), 2]] = scatter(x, dim=0, devices=[1, 2])
    wAs: Tuple[Tensor[(F, H), 1], Tensor[(F, H), 2]] = broadcast(x, devices=[1, 2])
    as: Tuple[Tensor[(B/N, H), 1], Tensor[(B/N, H), 2]] = pmap(
        lambda d:
        lambda (xi: Tensor[(B/N, F), d], wAi: Tensor[(F, H), d]): {
            ai: Tensor[(B/N, H), d] = MatMul(xi, wAi)
            return ai
        },
        (xs, wAs)
    )
    a: Tensor[(B, H), 0] = gather(as, dim=0, to_device=0)
    return a
```

First, run a data-parallel transform on it:

```python
def matmul_dp(wA: Tensor[(F, H), 0], x: Tensor[(B, F), 0]):
    xs: Tuple[Tensor[(B/N, F), 1], Tensor[(B/N, F), 2]] = scatter(x, dim=0, devices=[1, 2])
    wAs: Tuple[Tensor[(F, H), 1], Tensor[(F, H), 2]] = broadcast(x, devices=[1, 2])
    as: Tuple[Tensor[(B/N, H), 1], Tensor[(B/N, H), 2]] = pmap(
        lambda d:
        lambda (xi: Tensor[(B/N, F), d], wAi: Tensor[(F, H), d]): {
            ai: Tensor[(B/N, H), d] = MatMul(xi, wAi)
            return ai
        },
        (xs, wAs)
    )
    a: Tensor[(B, H), 0] = gather(as, dim=0, to_device=0)
    return a
```

If we now run a horizontal parallel transform on the body of the pmap, how do we represent the result?

## Subdevices: d, d.1, d.2, etc

Problem: how do we figure out which concrete device to map d.1 to?

```python
def nested_pmap_tuple(wA: Tensor[(F, H), 0], x: Tensor[(B, F), 0]):
    xs: Tuple[Tensor[(B/2, F), 1], Tensor[(B/2, F), 2]] = scatter(x, dim=0, devices=[1, 2])
    wAs: Tuple[Tensor[(F, H), 1], Tensor[(F, H), 2]] = broadcast(x, devices=[1, 2])
    as: Tuple[Tensor[(B/2, H), 1], Tensor[(B/2, H), 2]] = pmap(
        lambda d:
        lambda (xi: Tensor[(B/2, F), d], wAi: Tensor[(F, H), d]): {
            xis: Tuple[Tensor[(B/2, F), d], Tensor[(B/2, F), d.1]] = broadcast(x, devices=[d, d.1])
            wAis: Tuple[Tensor[(F, H/2), d], Tensor[(F, H/2), d.1]] = scatter(wA, dim=1, devices=[d, d.1])
            ais: Tuple[Tensor[(B/2, H/2), d], Tensor[(B/2, H/2), d.1]] = pmap(
                lambda dd:
                lambda (xij: Tensor[(B/2, F), dd], wAij: Tensor[(F, H/2), dd]): {
                    aij: Tensor[(B/2, H/2)] = Matmul(xij, wAij)
                    return aij
                },
                (xis, wAis)
            )
            ai: Tensor[(B/2, H), d] = gather(ais, dim=1, to_device=d)
            return ai
        },
        device_map={1: [1, 3], 2: [2, 4]},  # this tells simulator what d.1 is
        (xs, wAs)
    )
    a: Tensor[(B, H), 0] = gather(as, dim=0, to_device=0)
    return a
```

## device maps

```python
def nested_pmap_tuple(wA: Tensor[(F, H), 0], x: Tensor[(B, F), 0]):
    device_map = {
        1: [1, 3],
        2: [2, 4],  # etc
    }
    xs: Tuple[Tensor[(B/2, F), 1], Tensor[(B/2, F), 2]] = scatter(x, dim=0, devices=device_map.keys())
    wAs: Tuple[Tensor[(F, H), 1], Tensor[(F, H), 2]] = broadcast(x, devices=device_map.keys())
    as: Tuple[Tensor[(B/2, H), 1], Tensor[(B/2, H), 2]] = pmap(
        lambda d:
        lambda (xi: Tensor[(B/2, F), d], wAi: Tensor[(F, H), d]): {
            xis: Tuple[Tensor[(B/2, F), ?], Tensor[(B/2, F), ?]] = broadcast(x, devices=device_map[d])
            wAis: Tuple[Tensor[(F, H/2), ?], Tensor[(F, H/2), ?]] = scatter(wA, dim=1, devices=device_map[d])
            ais: Tuple[Tensor[(B/2, H/2), ?], Tensor[(B/2, H/2), ?]] = pmap(
                lambda dd:
                lambda (xij: Tensor[(B/2, F), dd], wAij: Tensor[(F, H/2), dd]): {
                    aij: Tensor[(B/2, H/2)] = Matmul(xij, wAij)
                    return aij
                },
                (xis, wAis)
            )
            ai: Tensor[(B/2, H), d] = gather(ais, dim=1, to_device=d)
            return ai
        },
        (xs, wAs)
    )
    a: Tensor[(B, H), 0] = gather(as, dim=0, to_device=0)
    return a
```

The `?` are because there's no way to annotate those values with a `Device`.
Don't have to annotate types with devices before simulator!
Simulator could simultaneously do type/device propagation using the concrete device values from something like device_map.

## d2 = get_sub_device(d)

Again, not sure how to resolve the subdevices to concrete devices before simulation
I prefer all intelligent choices being explicit in the IR

## Device arithmetic

But then we'd need to do verification to ensure devices in inner pmap are disjoint.

```python
# Use device arithmetic -- hard to ensure disjoint devices
def nested_pmap(wA: Tensor[(F, H), 0], x: Tensor[(B, F), 0]):
    device_tree = (...)
    xs: Tuple[Tensor[(B/2, F), 1], Tensor[(B/2, F), 3]] = scatter(x, dim=0, devices=[1, 3])
    wAs: Tuple[Tensor[(F, H), 1], Tensor[(F, H), 3]] = broadcast(x, devices=[1, 3])
    as: Tuple[Tensor[(B/2, H), 1], Tensor[(B/2, H), 3]] = pmap(
        lambda d:
        lambda (xi: Tensor[(B/2, F), d], wAi: Tensor[(F, H), d]): {
            xis: Tuple[Tensor[(B/2, F), d], Tensor[(B/2, F), d+1]] = broadcast(x, devices=[d, d+1])
            wAis: Tuple[Tensor[(F, H/2), d], Tensor[(F, H/2), d+1]] = scatter(wA, dim=1, devices=[d, d+1])
            ais: Tuple[Tensor[(B/2, H/2), d], Tensor[(B/2, H/2), d+1]] = pmap(
                lambda dd:
                lambda (xij: Tensor[(B/2, F), dd], wAij: Tensor[(F, H/2), dd]): {
                    aij: Tensor[(B/2, H/2)] = Matmul(xij, wAij)
                    return aij
                },
                (xis, wAis)
            )
            ai: Tensor[(B/2, H), d] = gather(ais, dim=1, to_device=d)
            return ai
        },
        (xs, wAs)
    )
    a: Tensor[(B, H), 0] = gather(as, dim=0, to_device=0)
    return a
```

## Misc. Thoughts

It would be nice to have concrete devices --- no pmap?
This example can be done with only a single pmap, but it's probably too hard to write the transforms to do the analysis and keep pmap count at 1.
Easy to write smaller simpler transforms and compose them

## DeviceMap implementation proposal

```
Python
@dataclass(frozen=True)
class DeviceMap(Type):
    
    devices: Dict[Device, Union[DeviceMap, List[Device]]]
    
    def __getitem__(self, key: Device):
        return self.devices[key]
 ```

The `DeviceMap` type serves as a wrapper over a Python dict with `Device` keys and either another `DeviceMap` or a list of `Device`s as values.
This enables representation of `Device` trees with arbitrary depth, where internal nodes are `DeviceMap`s and leaf nodes are the final list of `Device`s.
We add a `DeviceMap` value as an input to the function, initialized with the actual `Device` tree.

When we apply a transform to the graph, we specify a list of `Device` keys into the `DeviceMap`, with each key corresponding to a successive depth in
the device tree. These keys will be stored as attributes on the `Broadcast` and `Scatter` ops. The keys can either be explicit `Device`s or `Device` variables.

The full explicit `Device` list will be resolved at simulation time. When the simulator reaches a `Scatter` or `Broadcast` op, it will index into the
`DeviceMap` using the op's `Device` key list to determine the list of output devices over which to distribute the input value.
An empty key list indicates that the top-level `DeviceMap` dictionary keys should be used as the list of output devices.
Note that we can infer the list of devices over which the `pmap` consumer op is bound to based on this list of output devices. 

If a key in the `Broadcast`/`Scatter` key list is a `Device` variable, then we know the op is within a `pmap` context. In this case we would know the
full list of `Device`s the enclosing `pmap` is bound to, and therefore we can substitute the `Device` variable key with these bound devices.
This can be extended to `pmap` subfunctions of arbitrary depth.

As an example, suppose we have the following `DeviceMap`:
```
Python
{
    1: [1, 3],
    2: [2, 4]
}
```
We first apply a data parallel transform, passing an empty `Device` key list. This produces the following function:
```
Python
def matmul(wA: Tensor[(F, H), 0], wB: Tensor[(H, C), 0], x: Tensor[(B, F), 0]):
    device_map: DeviceMap = {1: [1, 3], 2: [2, 4]}
    xs_dp: Tuple[Tensor[(B/2, F), 1], Tensor[(B/2, F), 2]] = scatter(x, dim=0, device_keys=[])
    wAs_dp: Tuple[Tensor[(F, H), 1], Tensor[(F, H), 2]] = broadcast(wA, device_keys=[])
    wBs_dp: Tuple[Tensor[(H, C), 1], Tensor[(H, C), 2]] = broadcast(wB, device_keys=[])
    ys_dp: Tuple[Tensor[(B/2, C), 1], Tensor[(B/2, C), 2]] = pmap(
        lambda d:
        lambda (xi: Tensor[(B/2, F), d], wAi: Tensor[(F, H), d], wBi: Tensor[(H, C), d]): {
            ai: Tensor[(B/2, H), d] = MatMul(xi, wAi)
            yi: Tensor[(B/2, C), d] = MatMul(ai, wBi)
            return yi
        },
        (xs_dp, wAs_dp, wBs_dp)
    )
    y: Tensor[(B, C), 0] = gather(ys_dp, dim=0, to_device=0)
    return y
```
Next we apply a horizontal parallel transform passing a `Device` key list of `[d]`, which gives us this function:
```
Python
def matmul(wA: Tensor[(F, H), 0], wB: Tensor[(H, C), 0], x: Tensor[(B, F), 0]):
    device_map: DeviceMap = {1: [1, 3], 2: [2, 4]}
    xs_dp: Tuple[Tensor[(B/2, F), ?], Tensor[(B/2, F), ?]] = scatter(x, dim=0, device_keys=[])
    wAs_dp: Tuple[Tensor[(F, H), ?], Tensor[(F, H), ?]] = broadcast(wA, device_keys=[])
    wBs_dp: Tuple[Tensor[(H, C), ?], Tensor[(H, C), ?]] = broadcast(wB, device_keys=[])
    ys_dp: Tuple[Tensor[(B/2, C), ?], Tensor[(B/2, C), ?]] = pmap(
        lambda d:
        lambda (xi: Tensor[(B/N, F), d], wAi: Tensor[(F, H), d], wBi: Tensor[(H, C), d]): {
            xs_hp: Tuple[Tensor[(B/2, F), ?], Tensor[(B/2, F), ?]] = broadcast(x, device_keys=[d])
            wAs_hp: Tuple[Tensor[(F, H/2), ?], Tensor[(F, H/2), ?]] = scatter(wA, dim=1, device_keys=[d])
            wBs_hp: Tuple[Tensor[(H/2, C), ?], Tensor[(H/2, C), ?]] = scatter(wB, dim=0, device_keys=[d])
            ys_hp: Tuple[Tensor[(B/2, C), ?], Tensor[(B/2, C), ?]] = pmap(
                lambda dd:
                lambda (xi: Tensor[(B/2, F), dd], wAi: Tensor[(F, H/2), dd], wBi: Tensor[(H/2, C), dd]): {
                    ai_hp: Tensor[(B/2, H/2), ?] = MatMul(xi, wAi)
                    yi_hp: Tensor[(B/2, C), ?] = MatMul(ai, wBi)
                    return yi_hp
                },
                (xs_hp, wAs_hp, wBs_hp)
            )
            y_hp: Tuple[Tensor[(B/2, C), ?], Tensor[(B/2, C), ?] = allreduce(ys_hp)
            return y_hp
        },
        (xs_dp, wAs_dp, wBs_dp)
    )
    y: Tensor[(B, C), 0] = gather(ys_dp, dim=0, to_device=0)
    return y
```
We can then fully resolve this graph's placement at simulation time.
