from dist_ir.ir import Module
from dist_ir.ir.type import Tensor, Float


def test_module_view():
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

    module_view = module.get_view(("Add0", "Add1", "Add2"))
    module_view_inputs = module_view.get_inputs()
    module_view_outputs = module_view.get_outputs()
    assert [v.name for v in module_view_inputs] == ["x0", "x1", "x2", "x3"]
    assert [v.name for v in module_view_outputs] == ["a2"]

    module_view = module.get_view(("Add3", "Add4", "Add5"))
    module_view_inputs = module_view.get_inputs()
    module_view_outputs = module_view.get_outputs()
    assert [v.name for v in module_view_inputs] == ["a2", "x4", "x5", "x6"]
    assert [v.name for v in module_view_outputs] == ["a5"]

    module_view = module.get_view(("Add6", "Add7", "Add8"))
    module_view_inputs = module_view.get_inputs()
    module_view_outputs = module_view.get_outputs()
    assert [v.name for v in module_view_inputs] == ["a5", "x7", "x8", "x9"]
    assert [v.name for v in module_view_outputs] == ["a8"]


if __name__ == "__main__":
    test_module_view()
