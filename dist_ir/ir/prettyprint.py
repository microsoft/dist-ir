"""
Pretty printer for DistIR. Uses the prettyprinter package, which uses a modified
Wadler-Leijen layout algorithm:
http://homepages.inf.ed.ac.uk/wadler/papers/prettier/prettier.pdf

Use cpprint on any DistIR object to print pretty-printed output, pformat to get
an str.
"""

from itertools import repeat

# import cpprint and pformat so clients don't need to import both prettyprint and prettyprinter
from prettyprinter import (  # pylint: disable=unused-import
    register_pretty,
    cpprint,
    pformat,
    set_default_style,
    pretty_call,
)

# Wadler constructors
from prettyprinter.doc import (
    # flat_choice,
    annotate,  # annotations affect syntax coloring
    concat,
    group,  # make what's in here a single line if enough space
    nest,  # newlines within here are followed by indent + arg
    # align,
    # hang,
    # NIL,
    LINE,  # Space or Newline
    SOFTLINE,  # nothing or newline
    HARDLINE,  # newline
)
from prettyprinter.prettyprinter import (
    Token,
    ASSIGN_OP,
    COMMA,
    COLON,
    LPAREN,
    RPAREN,
    LBRACE,
    RBRACE,
    LBRACKET,
    RBRACKET,
    pretty_dispatch,
)
from prettyprinter.utils import intersperse

from .function import Function, FunctionMaker
from .value import Value
from .type import Type, Int32, Int64, Float32, Tensor, TupleType
from .device import Device
from .op import Op


# Set default style to light, as it's readable on both light/dark backgrounds
set_default_style("light")


def _join(*args):
    """Join the docs `args` by putting COMMA, LINE in between every two
    consecutive docs. E.g.
        join(a, b, c) -> [a, COMMA, LINE, b, COMMA, LINE, c, COMMA, LINE]
    """
    res = [None] * (len(args) * 3 - 2)
    res[0::3] = args
    res[1::3] = repeat(COMMA, len(args) - 1)
    res[2::3] = repeat(LINE, len(args) - 1)
    return res


# These are primarily to enable syntax highlighting --
# it would otherwise be fine to just use the string/doc "s"
def pp_reserved(s):
    return annotate(Token.NAME_BUILTIN, s)


def pp_fnname(s):
    return annotate(Token.NAME_FUNCTION, s)


def pp_var(s):
    return annotate(Token.NAME_VARIABLE, s)


def pp_type(s):
    return annotate(Token.COMMENT_SINGLE, s)


def interline(*docs):
    return concat(intersperse(HARDLINE, docs))


# ----------------------------------------
# Pretty printer for each class:
# ----------------------------------------


def _pprint_function_body(function: Function, ctx):
    ops = [pretty_dispatch(op, ctx) for op in function.ops]
    # Include the outputs as a final "return" op
    outputs = concat(_join(*(r.name for r in function.outputs)))
    return_line = group(
        nest(ctx.indent, concat([pp_reserved("return"), LINE, outputs]))
    )
    ops.append(return_line)
    return ops


@register_pretty(Function)
def _(function: Function, ctx):
    ops = _pprint_function_body(function, ctx)
    return concat(
        [
            annotate(Token.KEYWORD_CONSTANT, "function "),
            pretty_call(ctx, pp_fnname(function.name), *function.inputs),
            nest(ctx.indent, concat([COLON, HARDLINE, interline(*ops)])),
        ]
    )


@register_pretty(FunctionMaker)
def _(function: FunctionMaker, ctx):
    ops = _pprint_function_body(function, ctx)
    return concat(
        [
            annotate(Token.KEYWORD_CONSTANT, "function* "),
            pretty_call(ctx, pp_fnname(function.name), *function.inputs),
            nest(ctx.indent, concat([COLON, HARDLINE, interline(*ops)])),
        ]
    )


@register_pretty(Op)
def _(op: Op, ctx):
    results = concat(_join(*(pretty_dispatch(r, ctx) for r in op.outputs)))
    args = concat(_join(*(v.name for v in op.inputs)))

    if op.op_type == "Pmap":
        lambda_body = _pprint_function_body(op.subfunctions[0], ctx)
        lambda_body = interline(*lambda_body)
        actual_args = group(
            concat(
                [
                    LPAREN,
                    nest(ctx.indent, concat([SOFTLINE, args])),
                    RPAREN,
                ]
            )
        )
        # TODO: Also print out the list of devices this pmaps over
        if isinstance(op.attributes["device_var"], str):
            d = op.attributes["device_var"]
        elif isinstance(op.attributes["device_var"], Device):
            d = str(op.attributes["device_var"].device_id)
        else:
            raise ValueError(
                f'op.attributes["device_var"] has '
                f'unknown type {type(op.attributes["device_var"])}'
            )
        pmap_args = nest(
            ctx.indent,
            concat(
                [
                    HARDLINE,
                    pp_reserved("lambda"),
                    " ",
                    d,
                    COLON,
                    HARDLINE,
                    pp_reserved("lambda"),
                    pretty_call(ctx, " ", *op.subfunctions[0].inputs),
                    COLON,
                    " ",
                    LBRACE,
                    nest(ctx.indent, concat([HARDLINE, lambda_body])),
                    HARDLINE,
                    RBRACE,
                    COMMA,
                    HARDLINE,
                    actual_args,
                    HARDLINE,
                ]
            ),
        )
        opcall = group(
            concat(
                [
                    pp_reserved("pmap"),
                    LPAREN,
                    pmap_args,
                    HARDLINE,
                    RPAREN,
                ]
            )
        )

        return group(
            nest(
                ctx.indent,
                concat([results, LINE, ASSIGN_OP, " ", opcall]),
            )
        )

    opcall = group(
        concat(
            [
                pp_fnname(op.op_type),
                LPAREN,
                nest(ctx.indent, concat([SOFTLINE, args])),
                SOFTLINE,
                RPAREN,
            ]
        )
    )
    return group(nest(ctx.indent, concat([results, LINE, ASSIGN_OP, " ", opcall])))


@register_pretty(Value)
def _(val, ctx):
    return group(
        concat([pp_var(val.name), COLON, LINE, pretty_dispatch(val.type, ctx)])
    )


@register_pretty(Type)
def _(typ, ctx):
    return pp_type(str(typ))  # Use str for any type not defined below


@register_pretty(Tensor)
def _(typ, ctx):
    shape = pretty_dispatch(typ.shape, ctx)
    dtype = pretty_dispatch(typ.dtype, ctx)
    dev = pretty_dispatch(typ.device, ctx)
    return group(
        concat([pp_type("Tensor"), LBRACKET] + _join(shape, dtype, dev) + [RBRACKET])
    )


@register_pretty(TupleType)
def _(typ, ctx):
    elem_types = _join(*(pretty_dispatch(t, ctx) for t in typ.types))
    return group(concat([pp_type("Tuple"), LBRACKET] + elem_types + [RBRACKET]))


@register_pretty(Device)
def _(d, ctx):
    return pp_reserved(str(d.device_id))
