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
    LBRACKET,
    RBRACKET,
    pretty_dispatch,
)
from prettyprinter.utils import intersperse

from .module import Module
from .value import Value
from .type import Type, Int, Float, Tensor, TupleType
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
    return concat(intersperse(LINE, docs))


# ----------------------------------------
# Pretty printer for each class:
# ----------------------------------------


@register_pretty(Module)
def _(module: Module, ctx):
    ops = [pretty_dispatch(op, ctx) for op in module.get_ops().values()]
    # Include the outputs as a final "return" op
    outputs = concat(_join(*(r.name for r in module.get_outputs())))
    return_line = group(
        nest(ctx.indent, concat([pp_reserved("return"), LINE, outputs]))
    )
    ops.append(return_line)
    return concat(
        [
            pretty_call(ctx, pp_fnname("Module"), *module.get_inputs()),
            nest(ctx.indent, concat([COLON, HARDLINE, interline(*ops)])),
        ]
    )


@register_pretty(Op)
def _(op: Op, ctx):
    results = concat(_join(*(pretty_dispatch(r, ctx) for r in op.get_out_edges())))
    args = concat(_join(*(v.name for v in op.get_in_edges())))

    if op.op_type == "Pmap":
        return pp_reserved("Pmap")

    opcall = group(
        concat(
            [
                pp_fnname(op.name),
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
    return group(concat([pp_type("Tuple"), LBRACKET] + _join(*typ.types) + [RBRACKET]))


@register_pretty(Device)
def _(d, ctx):
    return pp_reserved(str(d.device_id))
