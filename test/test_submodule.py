from dist_ir.ir import Module
from dist_ir.ir.type import Tensor, Float


def test_submodule():
    module = Module()

    inputs = []
    outputs = []
    num_ops = 9
    for i in range(num_ops + 1):
        inputs.append(module.add_input_value(f"x{i}", Tensor(Float(), (4, 4))))
    for i in range(num_ops):
        if i == 0:
            input_values = inputs[:2]
        else:
            input_values = [outputs[-1], inputs[i + 1]]
        outputs.append(
            module.add_op("Add", f"Add{i}", inputs=input_values, output_names=[f"a{i}"])
        )
    module.finalize()

    submodule = module.get_submodule(("Add0", "Add1", "Add2"))
    submodule_inputs = submodule.get_inputs()
    submodule_outputs = submodule.get_outputs()
    assert [v.name for v in submodule_inputs] == ["x0", "x1", "x2", "x3"]
    assert [v.name for v in submodule_outputs] == ["a2"]

    submodule = module.get_submodule(("Add3", "Add4", "Add5"))
    submodule_inputs = submodule.get_inputs()
    submodule_outputs = submodule.get_outputs()
    assert [v.name for v in submodule_inputs] == ["a2", "x4", "x5", "x6"]
    assert [v.name for v in submodule_outputs] == ["a5"]

    submodule = module.get_submodule(("Add6", "Add7", "Add8"))
    submodule_inputs = submodule.get_inputs()
    submodule_outputs = submodule.get_outputs()
    assert [v.name for v in submodule_inputs] == ["a5", "x7", "x8", "x9"]
    assert [v.name for v in submodule_outputs] == ["a8"]


if __name__ == "__main__":
    test_submodule()
