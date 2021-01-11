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