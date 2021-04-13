from dist_ir.ir import FunctionMaker
from dist_ir.ir.type import Float32, Tensor


def mlp(batch_size, input_dim, hidden_dim, output_dim, num_hidden_layers, device):
    function = FunctionMaker(name="mlp")
    x = function.add_input_value(
        "x",
        Tensor(dtype=Float32(), shape=(batch_size, input_dim), device=device),
    )
    z = function.add_input_value(
        "z",
        Tensor(dtype=Float32(), shape=(batch_size, output_dim), device=device),
    )
    weights = []
    for i in range(num_hidden_layers - 1):
        w = function.add_input_value(
            f"w{chr(ord('A')+i)}",
            Tensor(dtype=Float32(), shape=(input_dim, hidden_dim), device=device),
        )
        weights.append(w)
    w = function.add_input_value(
        f"w{chr(ord('A')+i+1)}",
        Tensor(dtype=Float32(), shape=(hidden_dim, output_dim), device=device),
    )
    weights.append(w)

    a = x
    for i, weight in enumerate(weights):
        y = function.add_op("MatMul", inputs=[a, weight], output_names=[f"y{i}"])
        a = function.add_op("Relu", inputs=[y], output_names=[f"a{i}"])

    l = function.add_op(
        "Loss", inputs=[a, z], attributes={"N": batch_size}, output_names=["l"]
    )
    dl = function.add_op(
        "LossGrad",
        inputs=[a, z],
        attributes={"N": batch_size},
        output_names=["dl"],
    )

    dy = dl
    for i, weight in enumerate(weights[::-1]):
        i = len(weights) - i - 1
        da = function.add_op(
            "ReluGrad",
            inputs=[function.ops[2 * i + 1].inputs[0], dy],
            output_names=[f"da{i}"],
        )
        dy, dw = function.add_op(
            "MatMulGrad",
            inputs=[function.ops[2 * i].inputs[0], weights[i], da],
            output_names=[f"dy{i}", f"dw{chr(ord('A')+i)}"],
        )
    return function.finalize()


def mlp_inference(
    batch_size, input_dim, hidden_dim, output_dim, num_hidden_layers, device
):
    function = FunctionMaker(name="mlp")
    weights = []
    for i in range(num_hidden_layers - 1):
        w = function.add_input_value(
            f"w{chr(ord('A')+i)}",
            Tensor(dtype=Float(), shape=(input_dim, hidden_dim), device=device),
        )
        weights.append(w)
    w = function.add_input_value(
        f"w{chr(ord('A')+i+1)}",
        Tensor(dtype=Float(), shape=(hidden_dim, output_dim), device=device),
    )
    weights.append(w)
    x = function.add_input_value(
        "x",
        Tensor(dtype=Float(), shape=(batch_size, input_dim), device=device),
    )

    a = x
    for i, weight in enumerate(weights):
        y = function.add_op("MatMul", inputs=[a, weight], output_names=[f"y{i}"])
        a = function.add_op("Relu", inputs=[y], output_names=[f"a{i}"])

    return function.finalize()


def mlp_inference_dp(
    batch_size, input_dim, hidden_dim, output_dim, num_hidden_layers, devices
):
    num_devices = len(devices)
    assert batch_size % num_devices == 0
    function = FunctionMaker(name="mlp")
    weights = {}
    x = {}
    for d in devices:
        for i in range(num_hidden_layers - 1):
            weights[i, d] = function.add_input_value(
                f"w{chr(ord('A')+i)}_{d.device_id}",
                Tensor(dtype=Float(), shape=(input_dim, hidden_dim), device=d),
            )
        weights[num_hidden_layers - 1, d] = function.add_input_value(
            f"w{chr(ord('A')+i+1)}_{d.device_id}",
            Tensor(dtype=Float(), shape=(hidden_dim, output_dim), device=d),
        )
        x[d] = function.add_input_value(
            f"x_{d.device_id}",
            Tensor(
                dtype=Float(), shape=(batch_size // num_devices, input_dim), device=d
            ),
        )

    a = x
    for i in range(num_hidden_layers):
        for d in devices:
            y = function.add_op(
                "MatMul",
                inputs=[a[d], weights[i, d]],
                output_names=[f"y{i}_{d.device_id}"],
            )
            a[d] = function.add_op(
                "Relu", inputs=[y], output_names=[f"a{i}_{d.device_id}"]
            )

    return function.finalize()
