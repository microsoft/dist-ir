from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List, Union

# NOTE: Disabling to pass GitHub automated test
# import mlir
from ..ir import Function, FunctionMaker, Value
from ..ir.device import Device
from ..ir.type import Float, Int32, Int64, Tensor


@dataclass
class Context:
    "Parsing context."
    # Variable name counter
    var_number: int = 0
    # Maps device ID (int) / device var (str) -> Device object
    devices: Dict[Union[int, str], Any] = field(default_factory=dict)
    # Map from MLIR value str -> DistIR value
    # TODO check that this mapping works for ops that return multiple return values
    values: Dict[str, Value] = field(default_factory=dict)


def _make_fresh_var(context: Context):
    context.var_number += 1
    return f"%var{context.var_number}"


def _get_device(d: Union[int, str], context: Context) -> Device:
    """Given an integer device ID or string device variable, return the
    corresponding Device object or create one if none exists.
    """
    # TODO not sure how to figure out device type here
    dev_type = "gpu"
    if d not in context.devices:
        if isinstance(d, int):
            context.devices[d] = Device(d, dev_type)
        elif isinstance(d, str):
            context.devices[d] = Device.get_new_device_variable(dev_type)
        else:
            raise ValueError(f"Unexpected device {d}")
    return context.devices[d]


def _parse_type(mlir_type, context: Context):
    # Unfortunately, I can't inspect the MLIR type object, so parsing the string:
    dtype_map = {"f32": Float(), "i32": Int32(), "i64": Int64()}

    def parse_shape_dtype(shape_str):
        dims = shape_str.strip().split("x")
        return tuple(int(d) for d in dims[:-1]), dtype_map[dims[-1]]

    # tensor<4x8xf32>
    matches = re.match(r"tensor<([^>]*)>", str(mlir_type))
    if matches:
        shape, dtype = parse_shape_dtype(matches.group(1))
        return Tensor(dtype, shape)

    # !dist.tensor<4x8xf32, 0>
    matches = re.match(r"!dist.tensor<([^>]*)>", str(mlir_type))
    if matches:
        args = matches.group(1).split(",")
        assert len(args) == 2
        shape, dtype = parse_shape_dtype(args[0])
        device = _get_device(args[1].strip(), context)
        return Tensor(dtype, shape, device)

    # TODO handle tuple types
    raise ValueError(f"Unknown MLIR type {mlir_type}")


def _parse_function(mlir_region, context: Context = None) -> Function:
    """Creates a DistIR Function out of an MLIR region. The region must be either
    the single region in a function, or the sub-region of a pmap operator.
    """
    if context is None:
        context = Context(-1, {}, {})

    assert len(mlir_region.blocks) == 1
    entry_block = mlir_region.blocks[0]

    function = FunctionMaker()

    # Find the inputs
    for arg in entry_block.arguments:
        v = function.add_input_value(
            f"%arg{arg.arg_number}", _parse_type(arg.type, context)
        )
        assert str(arg) not in context.values
        context.values[str(arg)] = v

    returned = False
    for op in entry_block:
        if returned:
            raise ValueError(f"Unexpected op after function return:\n{op}")

        # Collect name and arguments
        op_name = op.operation.name
        args = [context.values[str(operand)] for operand in op.operands]

        # Collect attributes
        attributes = {}
        for attribute in op.attributes:
            # Parse values from string reprs since MLIR bindings insufficient:
            value = str(attribute.attr)
            if value.endswith(": i64"):
                value = int(value.split(":")[0].strip())
            elif value.startswith("[") and value.endswith("]"):
                value = json.loads(value)
            # All others are assumed to be strings

            # Need to convert the value of the devices attribute to a list of
            # DistIR device objects
            # TODO better place to do this?
            if attribute.name == "devices":
                assert isinstance(value, list)
                value = [_get_device(d, context) for d in value]

            attributes[attribute.name] = value

        # Create output names (TODO should be done by Op.__init__ or add_op)
        output_names = [_make_fresh_var(context) for _ in op.results]

        subfunctions = []

        if op_name == "std.return" or op_name == "dist.return":
            # Set return values as function outputs
            function.set_outputs(args)
            returned = True
            continue
        if op_name == "dist.pmap":
            op_name = "Pmap"
            # Create a new device and add it to the context.devices map
            new_device = Device.get_new_device_variable("gpu")
            assert attributes["device_var"] not in context.devices
            context.devices[attributes["device_var"]] = new_device
            # Parse the subfunction
            subfunctions.append(_parse_function(op.regions[0], context))
            # Remove device var from context.devices as it is only in scope for subfunction
            del context.devices[attributes["device_var"]]

        # Create an op and add it to the function
        outs = function.add_op(
            op_name,
            inputs=args,
            attributes=attributes,
            output_names=output_names,
            subfunctions=subfunctions,
        )
        if not isinstance(outs, tuple):
            outs = (outs,)
        assert len(outs) == len(op.results)

        # Collect the results and add to values
        for val, mlirval in zip(outs, op.results):
            assert str(mlirval) not in context.values
            context.values[str(mlirval)] = val

    return function.finalize()


def _parse_mlir_module(mlir_module) -> List[Function]:
    functions = []
    for func in mlir_module.body.operations:
        if str(func) == "module_terminator":
            break
        assert len(func.regions) == 1

        functions.append(_parse_function(func.regions[0]))
    return functions


def parse_mlir_str(mlir_str: str) -> List[Function]:
    ctx = mlir.ir.Context()
    ctx.allow_unregistered_dialects = True

    mlir_module = mlir.ir.Module.parse(mlir_str, ctx)
    return _parse_mlir_module(mlir_module)


def parse_mlir_file(filename) -> List[Function]:
    with open(filename, "r") as fin:
        mlir_str = fin.read()

    return parse_mlir_str(mlir_str)


# TODO things missing in MLIR Python bindings:
# - MLIR tensor type -> rank and dims and dtype
# - pointer from operand to operation that created it
# - hash for PyValue
# - Way to convert attribute values (e.g. i64s, arrays) to python values
