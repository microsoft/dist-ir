from dist_ir.ir import FunctionMaker
from dist_ir.ir.type import Tensor, Float


def test_subfunction():
    function = FunctionMaker()

    inputs = []
    outputs = []
    num_ops = 9
    for i in range(num_ops + 1):
        inputs.append(function.add_input_value(f"x{i}", Tensor(Float(), (4, 4))))
    for i in range(num_ops):
        if i == 0:
            input_values = inputs[:2]
        else:
            input_values = [outputs[-1], inputs[i + 1]]
        outputs.append(
            function.add_op(
                "Add", f"Add{i}", inputs=input_values, output_names=[f"a{i}"]
            )
        )
    function = function.finalize()

    subfunction = function.get_subfunction(function.ops[:3])
    subfunction_inputs = subfunction.inputs
    subfunction_outputs = subfunction.outputs
    assert [v.name for v in subfunction_inputs] == ["x0", "x1", "x2", "x3"]
    assert [v.name for v in subfunction_outputs] == ["a2"]

    subfunction = function.get_subfunction(function.ops[3:6])
    subfunction_inputs = subfunction.inputs
    subfunction_outputs = subfunction.outputs
    assert [v.name for v in subfunction_inputs] == ["a2", "x4", "x5", "x6"]
    assert [v.name for v in subfunction_outputs] == ["a5"]

    subfunction = function.get_subfunction(function.ops[6:])
    subfunction_inputs = subfunction.inputs
    subfunction_outputs = subfunction.outputs
    assert [v.name for v in subfunction_inputs] == ["a5", "x7", "x8", "x9"]
    assert [v.name for v in subfunction_outputs] == ["a8"]


if __name__ == "__main__":
    test_subfunction()
