import numpy as np

from dist_ir.ir import Device, FunctionMaker, cpprint
from dist_ir.ir.type import Float, Tensor, TupleType
from dist_ir.executor import SequentialExecutor


class Helper:
    def __init__(self, batch_size=256, input_dim=256, hidden_dim=256, output_dim=256):
        self.devices = [Device(i, "gpu") for i in range(5)]
        self.executor = SequentialExecutor("numpy")
        self.function = FunctionMaker()
        self.x = self.function.add_input_value(
            "x", Tensor(Float(), (batch_size, input_dim), device=self.devices[0])
        )
        self.wA = self.function.add_input_value(
            "wA", Tensor(Float(), (input_dim, hidden_dim), device=self.devices[0])
        )
        self.wB = self.function.add_input_value(
            "wB", Tensor(Float(), (hidden_dim, output_dim), device=self.devices[0])
        )
        _x = np.random.normal(size=(batch_size, input_dim))
        _wA = np.random.normal(size=(input_dim, hidden_dim))
        _wB = np.random.normal(size=(hidden_dim, output_dim))
        self.input_data = {
            self.x: _x,
            self.wA: _wA,
            self.wB: _wB,
        }


def test_data_parallelism_followed_by_horizontal_parallelism():
    h = Helper()
    xs = h.function.add_op(
        "Scatter",
        "Scatter/x",
        inputs=[h.x],
        attributes={"dim": 0, "devices": h.devices[1:]},
        output_names=["xs"],
    )
    wAs = h.function.add_op(
        "Broadcast",
        "Broadcast/wA",
        inputs=[h.wA],
        attributes={"devices": h.devices[1:]},
        output_names=["wAs"],
    )
    wBs = h.function.add_op(
        "Scatter",
        "Scatter/wB",
        inputs=[h.wB],
        attributes={"dim": 1, "devices": h.devices[1:]},
        output_names=["wBs"],
    )
    a_subfunction = FunctionMaker()
    a_subfunction.inputs.append(h.x)
    a_subfunction.inputs.append(h.wA)
    ai = a_subfunction.add_op(
        "MatMul", "MatMul/a", inputs=[h.x, h.wA], output_names=["a"]
    )
    a_subfunction = a_subfunction.finalize()
    ais = h.function.add_op(
        "Pmap",
        "Pmap/a",
        inputs=[xs, wAs],
        subfunctions=[a_subfunction],
        attributes={"devices": h.devices[1:]},
        output_names=["ais"],
    )
    as_ = h.function.add_op(
        "Allgather",
        "Allgather/ais",
        inputs=[ais],
        attributes={"dim": 0, "devices": h.devices[1:]},
        output_names=["as"],
    )
    y_subfunction = FunctionMaker()
    y_subfunction.inputs.append(ai)
    y_subfunction.inputs.append(h.wB)
    yi = y_subfunction.add_op(
        "MatMul", "MatMul/y", inputs=[ai, h.wB], output_names=["y"]
    )
    y_subfunction = y_subfunction.finalize()
    yis = h.function.add_op(
        "Pmap",
        "Pmap/y",
        inputs=[as_, wBs],
        subfunctions=[a_subfunction],
        attributes={"devices": h.devices[1:]},
        output_names=["yis"],
    )
    y = h.function.add_op(
        "Gather",
        "Gather/y",
        inputs=[yis],
        attributes={"dim": 1, "device": h.devices[0]},
        output_names=["y"],
    )
    h.function = h.function.finalize()
    output_data = h.executor.compute(h.function, h.input_data)
    y = output_data[y]
    ref = np.matmul(
        np.matmul(h.input_data[h.x], h.input_data[h.wA]), h.input_data[h.wB]
    )
    np.testing.assert_array_almost_equal(y, ref)


if __name__ == "__main__":
    test_data_parallelism_followed_by_horizontal_parallelism()
