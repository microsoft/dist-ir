# Composable Transformation Examples

This doc presents examples exploring the joint space of D/H/P parallelism for a simple 2-layer toy model.

### Original function:
```Python
def mlp(
    wA: Tensor[(F, H), 0], wB: Tensor[(H, C), 0], x: Tensor[(B, F), 0]
):
    a: Tensor[(B, H), 0] = MatMul(x, wA)
    y: Tensor[(B, C), 0] = MatMul(a, wB)
    return y
```

## Single parallelism style examples

### Data parallelism:
```Python
def mlp(
    wA: Tensor[(F, H), 0], wB: Tensor[(H, C), 0], x: Tensor[(B, F), 0]
):
    xs: Tuple[Tensor[(B/N, F), 1], ..., Tensor[(B/N, F), N]] = scatter(x, dim=0, devices=[1..N])
    wAs: Tuple[Tensor[(F, H), 1], ..., Tensor[(F, H), N]] = broadcast(wA, devices=[1..N])
    wBs: Tuple[Tensor[(H, C), 1], ..., Tensor[(H, C), N]] = broadcast(wB, devices=[1..N])
    (
        yis: Tuple[Tensor[(B/N, C), 1], ..., Tensor[(B/N, C), N]],
    ) = pmap(
        device_var=d,
        fn=lambda (xi: Tensor[(B/N, F), d]), (wAi: Tensor[(F, H), d]), (wBi: Tensor[(H, C), d]): {
            ai: Tensor[(B/N, H), d] = MatMul(xi, wAi)
            yi: Tensor[(B/N, C), d] = MatMul(ai, wBi)
            return yi
        },
        (xs, wAs, wBs)
    )
    y: Tensor[(B, C), 0] = gather(yis, dim=0, device=0)
    return y
```
 
### Horizontal parallelism:
```Python
def mlp(
    wA: Tensor[(F, H), 0], wB: Tensor[(H, C), 0], x: Tensor[(B, F), 0]
):
    xs: Tuple[Tensor[(B, F), 1], ..., Tensor[(B, F), N]] = broadcast(x, devices=[1..N])
    wAs: Tuple[Tensor[(F, H/N), 1], ..., Tensor[(F, H/N), N]] = scatter(wA, dim=1, devices=[1..N])
    wBs: Tuple[Tensor[(H, C/N), 1], ..., Tensor[(H, C/N), N]] = scatter(wB, dim=0, devices=[1..N])
    (
        yis: Tuple[Tensor[(B, C/N), 1], ..., Tensor[(B, C/N), N]],
    ) = pmap(
        device_var=d,
        fn=lambda (xi: Tensor[(B, F), d]), (wAi: Tensor[(F, H/N), d]), (wBi: Tensor[(F/N, C), d]): {
            ai: Tensor[(B, H/N), d] = MatMul(xi, wAi)
            yi: Tensor[(B, C), d] = MatMul(ai, wBi)
            return yi
        },
        (xs, wAs, wBs)
    )
    ys: Tuple[Tensor[(B, C), 1], ..., Tensor[(B, C), N]] = allreduce(yis)
    return ys
```

### Pipeline parallelism
```Python
def mlp(
    wA: Tensor[(F, H), 0], wB: Tensor[(H, C), 0], x: Tensor[(B, F), 0]
):
    xs: Tuple[Tensor(B/K, F), 0], ..., Tensor[(B/K, F), 0]] = split(x, K)
    
    # TODO: Insert sends of xs, wA, wB
 
    # Timestep 0
    a0: Tensor[(B/K, H), 1] = MatMul(xs[0], wA)
    a0@2: Tensor[(B/K, H), 2] = send(a0, 2)
    
    # Timestep 1
    a1: Tensor[(B/K, H), 1] = MatMul(xs[1], wA)
    a1@2: Tensor[(B/K, H), 2] = send(a1, 2)
    y0: Tensor[(B/K, C), 2] = MatMul(a0@2, wB)
    
    # Timestep 3
    y1: Tensor[(B/K, C), 2] = MatMul(a1@2, wB)
    
    y: Tensor[(B, C), 2] = concat((y0, y1), dim=0)
    # TODO: Move y to device 0?
    return y
```

## Sequentially composed parallelism examples 

### Data parallelism followed by horizontal parallelism:
```Python
def mlp(
    wA: Tensor[(F, H), 0], wB: Tensor[(H, C), 0], x: Tensor[(B, F), 0]
):
    xs: Tuple[Tensor[(B, F), 1], ..., Tensor[(B, F), N]] = scatter(x, dim=0, devices=[1..N])
    wAs: Tuple[Tensor[(F, H), 1], ..., Tensor[(F, H), N]] = broadcast(wA, devices=[1..N])
    wBs: Tuple[Tensor[(H, C/N), 1], ..., Tensor[(H, C/N), N]] = scatter(wB, dim=1, devices=[1..N])
    (
        ais: Tuple[Tensor[(B/N, H), 1], ..., Tensor[(B/N, H), N]],
    ) = pmap(
        device_var=d,
        fn=lambda (xi: Tensor[(B/N, F), d]), (wAi: Tensor[(F, H), d])): {
            ai: Tensor[(B/N, H), d] = MatMul(xi, wAi)
            return ai
        },
        (xs, wAs)
    )
    as: Tuple[Tensor[(B, H), 1], ..., Tensor[(B, H), N]] = allgather(ais, dim=0)
    (
        yis: Tuple[Tensor[(B, C/N), 1], ..., Tensor[(B, C/N), N]],
    ) = pmap(
        device_var=d,
        fn=lambda (ai: Tensor[(B, H), d]), (wBi: Tensor[(H, C/N), d])): {
            yi: Tensor[(B, C/N), d] = MatMul(ai, wBi)
            return yi
        },
        (as, wBs)
    )
    y: Tensor[(B, C), 0] = gather(yis, dim=1, device=0)
    return y
```

### Horizontal parallelism followed by data parallelism:
```Python
def mlp(
    wA: Tensor[(F, H), 0], wB: Tensor[(H, C), 0], x: Tensor[(B, F), 0]
):
    xs: Tuple[Tensor[(B, F), 1], ..., Tensor[(B, F), N]] = broadcast(x, devices=[1..N])
    wAs: Tuple[Tensor[(F, H/N), 1], ..., Tensor[(F, H/N), N]] = scatter(wA, dim=1, devices=[1..N])
    wBs: Tuple[Tensor[(H, C), 1], ..., Tensor[(H, C), N]] = broadcast(wB, devices=[1..N])
    (
        ais: Tuple[Tensor[(B, H/N), 1], ..., Tensor[(B, H/N), N]],
    ) = pmap(
        device_var=d,
        fn=lambda (xi: Tensor[(B, F), d]), (wAi: Tensor[(F, H/N), d])): {
            ai: Tensor[(B, H/N), d] = MatMul(xi, wAi)
            return ai
        },
        (xs, wAs)
    )
    as: Tuple[Tensor[(B, H), 1], ..., Tensor[(B, H), N]] = allgather(ais, dim=1)
    (
        yis: Tuple[Tensor[(B/N, C), 1], ..., Tensor[(B/N, C), N]],
    ) = pmap(
        device_var=d,
        fn=lambda (ai: Tensor[(B/N, H), d]), (wBi: Tensor[(H, C), d])): {
            yi: Tensor[(B/N, C), d] = MatMul(ai, wBi)
            return yi
        },
        (as, wBs)
    )
    y: Tensor[(B, C), 0] = gather(yis, dim=0, device=0)
    return y
```

## Recursively composed parallelism examples 

### Data parallelism over horizontal parallelism:
```Python
def mlp(
    wA: Tensor[(F, H), 0], wB: Tensor[(H, C), 0], x: Tensor[(B, F), 0]
):
    xs_dp: Tuple[Tensor[(B/N, F), ?], ..., Tensor[(B/N, F), ?]] = scatter(x, dim=0, devices=[?..?])
    wAs_dp: Tuple[Tensor[(F, H), ?], ..., Tensor[(F, H), ?]] = broadcast(wA, devices=[?..?])
    wBs_dp: Tuple[Tensor[(H, C), ?], ..., Tensor[(H, C), ?]] = broadcast(wB, devices=[?..?])
    
    (
        yis_dp: Tuple[Tensor[(B/N, C), ?], ..., Tensor[(B/N, C), ?]]
    ) = pmap(
        device_var=d_dp,
        fn=lambda (xi_dp: Tensor[(B/N, F), d_dp]), (wAi_dp: Tensor[(F, H), d_dp]),
                  (wBi_dp: Tensor[(H, C), d_dp]): {
            xs_hp = broadcast(xi_dp, devices=[?..?])
            wAs_hp = scatter(wAi_dp, dim=1, devices=[?..?])
            wBs_hp = scatter(wBi_dp, dim=1, devices=[?..?])
            (
                ais: Tuple[Tensor[(B/N, H/N), ?], ..., Tensor[(B/N, H/N), ?]]
            ) = pmap(
                device_var=d_hp,
                fn = lambda (xi: Tensor[(B/N, F), d_hp]), (wAi_hp: Tensor[(F, H/N), d_hp]),
                            (wBi_hp: Tensor[(F/N, C), d_hp]): {
                    ai: Tensor[(B/N, H/N), d_hp] = MatMul(xi, wAi_hp)
                    yi: Tensor[(B/N, C), d_hp] = MatMul(ai, wBi_hp)
                },
                (as, wAs_hp, wBs_hp)
            )
            ys_hp: Tuple[Tensor[(B/N, C), ?], ..., Tensor[(B/N, C), ?]] = allreduce(yis_hp)
            # TODO: Select just one copy to return?
            return ys_hp
        },
        (xs_dp, wAs_dp, wBs_dp)
    )
    y: Tensor[(B, C), 0] = gather(yis_dp, dim=0, device=0)
    return y
```

### Horizontal parallelism over data parallelism
```Python
def mlp(
    wA: Tensor[(F, H), 0], wB: Tensor[(H, C), 0], x: Tensor[(B, F), 0]
):
    xs_hp: Tuple[Tensor[(B/N, F), 1], ..., Tensor[(B/N, F), N]] = broadcast(x, devices=[?..?])
    wAs_hp: Tuple[Tensor[(F, H/N), 1], ..., Tensor[(F, H/H), N]] = scatter(wA_hp, dim=1, devices=[?..?])
    wBs_hp: Tuple[Tensor[(H/N, C), 1], ..., Tensor[(H/N, C), N]] = scatter(wB_hp, dim=0, devices=[?..?])
    (
        yis_hp: Tuple[Tensor(B, C), ?], ..., Tensor[Tensor(B, C), ?]]
    ) = pmap(
        device_var=d_hp,
        fn=lambda (xi_hp: Tensor[(B, F), d_hp]), (wA_hp: Tensor[(F, H/N), d_hp]),
                  (wB_hp: Tensor[(H/N, C), d_hp]): {
            xs_dp: Tuple[Tensor[(B/N, F), 1], ..., Tensor[(B/N, F), N]] = scatter(xi_hp, dim=0, devices=[?..?])
            wAs_dp: Tuple[Tensor[(F, H/N), 1], ..., Tensor[(F, H/H), N]] = broadcast(wA_hp, devices=[?..?])
            wBs_dp: Tuple[Tensor[(H/N, C), 1], ..., Tensor[(H/N, C), N]] = broadcast(wB_hp, devices=[?..?])
            (
                yis_dp: Tuple[Tensor[(B/N, C), ?], ..., Tensor[(B/N, C), ?]],
            ) = pmap(
                device_var=d_dp,
                fn=lambda (xi_dp: Tensor[(B/N, F), d_dp]), (wAi_dp: Tensor[(F, H/N), d_dp]),
                          (wBi_dp: Tensor[(H/N, C), d_dp]): {
                    ai_dp: Tensor[(B/N, H/N), d_dp] = MatMul(xi, wAi)
                    yi_dp: Tensor[(B/N, C), d_dp] = MatMul(ai, wBi)
                    return yi_dp
                },
                (xs_dp, wAs_dp, wBs_dp)
            )
            y_dp: Tensor[(B, C), ?] = gather(yis_dp, dim=0, device=?)
            return y_dp
        },
        (xs_hp, wAs_hp, wBs_hp)
    )
    y_hp: Tensor[(B, C), ?] = allreduce(yis_hp)
    return y_hp
```

### Data parallelism over pipeline parallelism
```Python
def mlp(
    wA: Tensor[(F, H), 0], wB: Tensor[(H, C), 0], x: Tensor[(B, F), 0]
):
    xs: Tuple[Tensor[(B/N, F), 1], ..., Tensor[(B/N, F), N]] = scatter(x, dim=0, devices=[?..?])
    wAs: Tuple[Tensor[(F, H), 1], ..., Tensor[(F, H), N]] = broadcast(wA, devices=[?..?])
    wBs: Tuple[Tensor[(H, C), 1], ..., Tensor[(H, C), N]] = broadcast(wB, devices=[?..?])
    (
        yis: Tuple[Tensor[(B/N, C), ?], ..., Tensor[(B/N, C), ?]],
    ) = pmap(
        device_var=d,
        fn=lambda (xi: Tensor[(B/N, F), d]), (wAi: Tensor[(F, H), d]), (wBi: Tensor[(H, C), d]): {
            xis: Tuple[Tensor[(B/N/K, F), d], ..., Tensor[(B/N/K, F), d] = split(xi, K)
            
            # NOTE: Assume K = 2
            # Timestep 0
            ai_0: Tensor[(B/N/K, H), ?] = MatMul(xis[0], wAi)
            ai_0@d?: Tensor[(B/N/K, H), ?] = send(xis[0], ?)
            
            # Timestep 1
            ai_1: Tensor[(B/N/K, H), ?] = MatMul(xis[1], wAi)
            ai_1@d?: Tensor[(B/N/K, H), ?] = send(xis[0], ?)
            yi_0: Tensor[(B/N/K, C), ?] = MatMul(ai_0@d?, wBi)
            
            # Timestep 2
            yi_1: Tensor[(B/N/K, C), ?] = MatMul(ai_1@d?, wBi)
            
            yi: Tensor[(B/N, C), ?] = concat((yi_0, yi_1), dim=0) 
            return yi
        },
        (xs, wAs, wBs)
    )
    y: Tensor[(B, C), ?] = gather(yis, dim=0, device=?)
    return y
```

### Horizontal parallelism over pipeline parallelism
```Python
def mlp(
    wA: Tensor[(F, H), 0], wB: Tensor[(H, C), 0], x: Tensor[(B, F), 0]
):
    xs: Tuple[Tensor[(B, F), ?], ..., Tensor[(B, F), ?]] = broadcast(x, devices=[?..?])
    wAs: Tuple[Tensor[(F, H/N), ?], ..., Tensor[(F, H/N), N?]] = scatter(wA, dim=1, devices=[?..?])
    wBs: Tuple[Tensor[(H/N, C), ?], ..., Tensor[(H/N, C), N]] = broadcast(wB, dim=0, devices=[?..?])
    (
        yis: Tuple[Tensor[(B, C), ?], ..., Tensor[(B, C), ?]],
    ) = pmap(
        device_var=d,
        fn=lambda (xi: Tensor[(B, F), d]), (wAi: Tensor[(F, H/N), d]), (wBi: Tensor[(H/N, C), d]): {
            xis: Tuple[Tensor[(B/K, F), d], ..., Tensor[(B/K, F), d] = split(xi, K)
            
            # NOTE: Assume K = 2
            # Timestep 0
            ai_0: Tensor[(B/K, H/N), ?] = MatMul(xis[0], wAi)
            ai_0@d?: Tensor[(B/K, H/N), ?] = send(xis[0], ?)
            
            # Timestep 1
            ai_1: Tensor[(B/K, H/N), ?] = MatMul(xis[1], wAi)
            ai_1@d?: Tensor[(B/K, H/N), ?] = send(xis[0], ?)
            yi_0: Tensor[(B/K, C), ?] = MatMul(ai_0@d?, wBi)
            
            # Timestep 2
            yi_1: Tensor[(B/K, C), ?] = MatMul(ai_1@d?, wBi)
            
            yi: Tensor[(B, C), ?] = concat((yi_0, yi_1), dim=0) 
            return yi
        },
        (xs, wAs, wBs)
    )
    y: Tensor[(B, C), ?] = allreduce(yis)
    return y
```
