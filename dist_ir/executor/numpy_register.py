import numpy as np


def _handle_negative_axis(axis, tensor_rank):
    return axis + tensor_rank if axis < 0 else axis


def _size_helper(shape, start, end):
    size = shape[start]
    for i in range(start + 1, end):
        size *= shape[i]
    return size


def add(op, x, y):
    return np.add(x, y)


def bias_fast_gelu_grad_dx(op, dy, x, b):
    kAlpha = np.sqrt(np.pi) * np.sqrt(0.5)
    kGamma = 0.044715
    kBeta = kGamma * kAlpha * 3.0
    input_shape = x.shape
    bias_shape = b.shape
    x_cube = np.power(x, 3)
    tanh_result = kAlpha * np.tanh(x + kGamma * x_cube)
    sech_sqr_result = 1 - (tanh_result * tanh_result)
    dx = dy * (
        0.5 * (tanh_result + sech_sqr_result * (kAlpha * x + kBeta + x_cube) + 1)
    )
    assert dx.shape == input_shape
    return dx


def cast(op, x):
    proto_dtype = op.attributes["to"]
    dtype = {
        1: np.float32,
        6: np.int32,
        7: np.int64,
        9: bool,
    }[proto_dtype]
    return x.astype(dtype)


def concat(op, *xs):
    # TODO make variadic
    dim = op.attributes["axis"]
    return np.concatenate(xs, axis=dim)


def constant(op):
    v = op.attributes["value"]
    if v.shape == (1,):
        return v[0]
    else:
        return v


def constant_of_shape(op, x):
    if "value" in op.attributes:
        value = op.attributes["value"]
    else:
        value = 0.0
    return np.full(shape=x.astype(np.int32), fill_value=value)


def div(op, x, y):
    return np.divide(x, y)


def dropout(op, x, ratio, training_mode):
    if training_mode:
        scale = 1.0 / (1.0 - ratio)
        mask = np.random.randint(0, 2, size=x.shape)
        x = scale * mask * x
        return x, mask
    else:
        return x


def dropout_grad(op, dy, mask, ratio, extra_input=None):
    # TODO: Figure out what extra input is
    if ratio == 0:
        return dy
    else:
        return mask * dy / (1.0 - ratio)


def expand(op, x, y):
    return x * np.ones(y)


def fast_gelu(op, x, y):
    # https://github.com/hendrycks/GELUs
    # TODO: What should we do with y?
    return 1.0 / (1.0 + np.exp(-1.702 * x))


def gather(op, x, y):
    if "axis" in op.attributes:
        axis = op.attributes["axis"]
    else:
        axis = 0
    res = np.take(x, y.astype(np.int64), axis=axis)
    if res.shape == (1,):
        return res[0]
    return res


def gather_grad(op, shape, indices, grad):
    # https://github.com/microsoft/onnxruntime/blob/master/orttraining/orttraining/test/training_ops/cpu/tensor/gather_grad_op_test.cc#L18
    axis = _handle_negative_axis(op.attributes["axis"], len(shape))
    num_batches = _size_helper(shape, 0, axis)
    gather_dimension_size = shape[axis]
    num_gathered_per_index = _size_helper(shape, axis + 1, len(shape))
    output = np.zeros(shape)
    for batch_idx in range(num_batches):
        output_batch_offset = batch_idx * gather_dimension_size * num_gathered_per_index
        grad_batch_offset = batch_idx * len(indices) * num_gathered_per_index
        for i in range(len(indices)):
            grad_row_offset = grad_batch_offset + i * num_gathered_per_index
            output_row_offset = (
                output_batch_offset + indices[i] * num_gathered_per_index
            )
            for j in range(num_gathered_per_index):
                output[output_row_offset + j] += grad[grad_row_offset + j]
    return output
    """
    def _size_from_dimension(shape, start, end):
        size = shape[start]
        for i in range(start + 1, end):
            size *= shape[i]
        return size

    output = np.zeros(shape)
    block_size = _size_from_dimension(shape, axis + 1, len(shape))
    N = indices.size
    input_block_size = _size_from_dimension(shape, axis, len(shape))
    output_block_size = N * block_size
    indices_max = shape[axis]
    grad_size = grad.size

    assert grad_size % block_size == 0

    for i in range(0, grad_size, grad_size // block_size):
        for g in range(i, block_size):
            input_block_index = g / output_block_size
            block_offset = g % output_block_size
            indices_index = block_offset // block_size
            offset = block_offset % block_size
            if len(indices.shape) == 0:
                idx = np.expand_dims(indices, 0)[indices_index]
            else:
                idx = indices[indices_index]
            input_index = (
                input_block_index * input_block_size + idx * block_size + offset
            )
            output[input_index] += grad[g]

    return output
    """


def gather_nd(op, data, indices):
    batch_dims = op.attributes["batch_dims"]

    # https://github.com/onnx/onnx/blob/b1e0bc9a31eaefc2a9946182fbad939843534984/onnx/backend/test/case/node/gathernd.py#L15
    data_rank = len(data.shape)

    # Check input tensors' shape/rank condition
    assert indices.shape[-1] <= data_rank

    # The list of data/indice shape of batch_dims
    batch_dims_shape = []

    # The number of elements in the batch_dims for data/indice array
    batch_dims_size = 1

    # Check the shape of indice and data are identicial for batch dims.
    for i in range(batch_dims):
        batch_dims_shape.append(indices.shape[i])
        batch_dims_size *= indices.shape[i]

    # Compute output of the op as below

    # Compute shape of output array
    output_shape = (
        batch_dims_shape + list(indices.shape)[batch_dims:-1]
        if (indices.shape[-1] == data_rank - batch_dims)
        else batch_dims_shape
        + list(indices.shape)[batch_dims:-1]
        + list(data.shape)[batch_dims + indices.shape[-1] :]
    )

    # Placeholder for output data
    output_data_buffer = []

    # Flatten 'indices' to 2D array
    reshaped_indices = indices.reshape(batch_dims_size, -1, indices.shape[-1])

    # Flatten 'data' to array of shape (batch_dim_size, data.shape[batch_dimes:])
    reshaped_data = data.reshape((batch_dims_size,) + data.shape[batch_dims:])

    # gather each scalar value from 'data'
    for batch_dim in range(reshaped_indices.shape[0]):
        for outer_dim in range(reshaped_indices.shape[1]):
            gather_index = tuple(reshaped_indices[batch_dim][outer_dim])
            output_data_buffer.append(reshaped_data[(batch_dim,) + gather_index])
    return np.asarray(output_data_buffer, dtype=data.dtype).reshape(output_shape)


def gather_nd_grad(op, input_shape, indices_tensor, update_tensor):
    output_tensor = np.zeros(input_shape)
    # TODO: Finish implementing

    """
    batch_dims = op.attributes["batch_dims"]
    indices_shape = indices_tensor.shape
    last_indices_dimension = batch_dims + indices_shape[-1]
    assert last_indices_dimension <= len(input_shape)

    grad_size = update_tensor.ndim
    slice_size = element_count_per_slice
    for i in range(grad_size):
        slice_offset = slice_offsets[i // slice_size]
        j = i % slice_size
        output_tensor[slice_offset + j] += update_tensor[i]
    """
    return output_tensor


def identity(op, x):
    return x


def gemm(op, a, b, c):
    alpha = op.attributes["alpha"]
    beta = op.attributes["beta"]
    if "transA" in op.attributes and op.attributes["transA"]:
        a = a.T
    if "transB" in op.attributes and op.attributes["transB"]:
        b = b.T
    return np.matmul(alpha * a, beta * b) + c


def join(op, *xs):
    return tuple(xs)


def layer_norm(op, x, scale, beta):
    eps = 1e-5
    mean = np.mean(x)
    std = np.std(x)
    x = (x - mean) / (pow(std, 2) + eps) * scale + beta
    return x, mean, std


def layer_norm_grad(op, y_grad, x, scale, mean, inv_std_var):
    assert mean == np.mean(x)
    a = y_grad * (x - mean) * inv_std_var
    b = y_grad * scale * inv_std_var
    c = y_grad * scale * inv_std_var * (x - mean) * inv_std_var
    x_grad = b - (x - mean) * inv_std_var * np.mean(c, axis=0)
    bias_grad = np.sum(y_grad, axis=1)
    scale_grad = np.sum(a, axis=1)
    assert x_grad.shape == x.shape
    return x_grad, bias_grad, scale_grad


def loss(op, x, y):
    N = op.attributes["N"]
    return np.square(x - y) / N


def loss_grad(op, x, y):
    N = op.attributes["N"]
    return 2 * (x - y) / N


def matmul(op, x, y):
    return np.matmul(x, y)


def matmul_grad(op, x, y, dz):
    return (np.dot(dz, y.T), np.dot(x.T, dz))
    # return (np.dot(x, dz), np.dot(y, dz))


def relu(op, x):
    return np.maximum(x, 0)


def relu_grad(op, x, dy):
    # TODO: fix
    dx = np.zeros(dy.shape)
    dx[dy > 0] = 1
    return dx


def mpi_allgather(op, *xs):
    v = mpi_gather(op, *xs)
    return tuple(v for i in range(len(xs)))


def mpi_allreduce(op, *xs):
    # TODO: Add attribute for reduction operator
    sum_ = np.sum(xs, axis=0)
    return tuple(sum_ for i in range(len(xs)))


def mpi_broadcast(op, x):
    return tuple(x for _ in range(len(op.attributes["devices"])))


def mpi_gather(op, *xs):
    dim = op.attributes["axis"]
    return np.concatenate(xs, axis=dim)


def mpi_reduce(op, *xs):
    return np.sum(xs, axis=0)


def mul(op, x, y):
    return x * y


def reduce_all_l2(op, *xs):
    return np.sqrt(sum([np.linalg.norm(x) for x in xs]))


def reduce_mean(op, x):
    if "keepdims" in op.attributes:
        keepdims = op.attributes["keepdims"]
    else:
        keepdims = 1
    return np.mean(x, axis=tuple(op.attributes["axes"]), keepdims=keepdims)


def reduce_sum(op, x):
    if "keepdims" in op.attributes:
        keepdims = op.attributes["keepdims"]
    else:
        keepdims = 1
    return np.sum(x, axis=tuple(op.attributes["axes"]), keepdims=keepdims)


def relu(op, x):
    return np.maximum(x, 0)


def reshape(op, x, new_shape):
    new_shape = list(new_shape)
    for i in range(len(new_shape)):
        if new_shape[i] == 0:
            new_shape[i] = x.shape[i]
    return np.reshape(x, new_shape)


def select(op, xs):
    dim = op.attributes["dim"]
    return xs[dim]


def shape(op, x):
    return np.array(x.shape, dtype=np.int64)


def slice_conc(op, x, starts, ends, axes, steps=None):
    # TODO handle the other cases, e.g. negative indices
    if steps is None:
        steps = [1] * len(starts)
    elif isinstance(steps, np.int64):
        steps = [steps] * len(starts)
    else:
        assert len(steps) == len(starts)
    slices = {
        axis: slice(s, e, step) for (s, e, axis, step) in zip(starts, ends, axes, steps)
    }
    slices = tuple(slices.get(d, slice(None)) for d in range(x.ndim))
    return x[slices]


def softmax(op, x):
    axis = op.attributes["axis"]
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def softmax_grad(op, dY, Y):
    # https://github.com/tensorflow/tensorflow/blob/dec8e0b11f4f87693b67e125e67dfbc68d26c205/tensorflow/python/ops/nn_grad.py#L285
    sum_channels = np.sum(dY * Y, -1, keepdims=True)
    return (dY - sum_channels) * Y


def softmax_cross_entropy_loss(op, x, target):
    weight = None
    if "ignore_index" in op.attributes:
        ignore_index = op.attributes["ignore_index"]
    else:
        ignore_index = None
    reduction = op.attributes["reduction"]
    get_log_prob = True

    # https://github.com/onnx/onnx/blob/b1e0bc9a31eaefc2a9946182fbad939843534984/onnx/backend/test/case/node/softmaxcrossentropy.py#L15
    input_shape = x.shape
    if len(input_shape) == 1:
        raise RuntimeError("Unsupported shape")

    target_shape = target.shape
    N = input_shape[0]
    C = input_shape[1]

    # compute log_softmax
    max_x = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - max_x)
    p = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    inp = np.log(p)
    log_prob = None
    if get_log_prob is True:
        log_prob = np.copy(inp)

    # initialize the positional weights when required
    gather_weight = None
    if weight is not None:
        # setting mode='clip' to deal with ignore_index > C or < 0 cases.
        # when the target value is > C or < 0, it doesn't matter which value we are
        # taking in gather_weight, since it will be set to 0 in the following if-block
        gather_weight = np.take(weight, target, mode="clip")
        # set `ignore_index`'s loss weight to 0.
        # The loss tensor will be multiplied by this weight tensor,
        # so `ingore_index`'s loss value will be eliminated.
        if ignore_index is not None:
            gather_weight = np.where(target == ignore_index, 0, gather_weight).astype(
                dtype=np.float32
            )
    elif ignore_index is not None:
        gather_weight = np.where(target == ignore_index, 0, 1).astype(dtype=np.float32)

    # if input is 4-d and above, make it 3-d
    if len(input_shape) != 3:
        inp = inp.reshape((N, C, -1))
        target = target.reshape((N, -1))

    # Get a dimension from the reshaped input.
    # If the original input shape is [N, C, H, W],
    # the D here should be H * W because we reshape
    # [N, C, H, W] to [N, C, H * W].
    D = inp.shape[2]
    neg_gather_element_input = np.zeros((N, D), dtype=np.float32)
    for i in range(N):
        for d in range(D):
            if target[i][d] != ignore_index:
                neg_gather_element_input[i][d] = -inp[i][target[i][d]][d]

    loss = neg_gather_element_input

    # if the input was 4-d or above reshape to the right shape
    if len(input_shape) != 3:
        loss = loss.reshape(target_shape)

    # apply the weights when required
    if gather_weight is not None:
        loss = gather_weight * loss
        if reduction == "mean":
            loss = loss.sum() / gather_weight.sum()
            if get_log_prob is True:
                return loss, log_prob
            else:
                return loss

    if reduction == "mean":
        loss = np.mean(loss)
    elif reduction == "sum":
        loss = np.sum(loss)

    if get_log_prob is True:
        return loss, log_prob
    else:
        return loss


def softmax_cross_entropy_loss_grad(op, dy, log_prob, label, weight=None):
    # https://github.com/microsoft/onnxruntime/blob/master/orttraining/orttraining/training_ops/cpu/loss/softmax_cross_entropy_loss.cc
    def get_permuation_and_shape(ncd_to_ndc, tensor_shape, new_shape, permutations):
        new_shape.append(tensor_shape[0])
        permutations.append(0)
        if ncd_to_ndc:
            for i in range(2, len(tensor_shape)):
                new_shape.append(tensor_shape[i])
                permutations.append(i)
            new_shape.append(tensor_shape[1])
            permutations.append(1)
        else:
            new_shape.append(tensor_shape[-1])
            permutations.append(len(tensor_shape) - 1)
            for i in range(1, len(tensor_shape) - 1):
                new_shape.append(tensor_shape[i])
                permutations.append(i)

    if "ignore_index" in op.attributes:
        ignore_index = op.attributes["ignore_index"]
    else:
        ignore_index = None
    reduction = op.attributes["reduction"]
    probability_shape = log_prob.shape
    d_logit = np.zeros(probability_shape)
    label_shape = label.shape
    n_d = len(label_shape)
    assert len(log_prob.shape) % n_d == 0
    c = len(log_prob.shape) // n_d
    new_shape = []
    permutations = []

    if len(probability_shape) > 2:
        get_permutation_and_shape(True, probability_shape, new_shape, permutations)
        tranpose_output = np.transpose(log_prob, permutations)
        log_prob = transpose_output

    if weight is not None:
        if reduction is None:
            for i in range(n_d):
                label_sample = label[i]
                weight_sample = weight_data[label_sample] * dy[i]
                for j in range(c):
                    index = i * c + j
                    if ignore_index == label_sample:
                        d_logit[index] = 0
                    else:
                        d_logit[index] = (
                            np.exp(log_prob[index])
                            - (label_sample == j) * weight_sample
                        )
        else:
            dy_scaled = dy
            if reduction == "mean":
                sum_weight = 0
                for i in range(0, n_d):
                    if ignore_index != label_data[i]:
                        sum_weight += weight[label[i]]
                if sum_weight != 0:
                    dy_scaled = dy_data / sum_weight

            for i in range(n_d):
                label_sample = label[i]
                weight_sample = weight[label_sample] * dy_scaled
                for j in range(c):
                    index = i * c + j
                    if ignore_index == label_sample:
                        d_logit[index] = 0
                    else:
                        d_logit[index] = (
                            np.exp(log_prob[index])
                            - (label_sample == j) * weight_sample
                        )
    else:
        if reduction is None:
            for i in range(n_d):
                label_sample = label[i]
                for j in range(c):
                    index = i * c + j
                    if ignore_index == label_sample:
                        d_logit[index] = 0
                    else:
                        d_logit[index] = (
                            np.exp(log_prob[index]) - (label_sample == j) * dy[i]
                        )
        else:
            dy_scaled = dy
            unignored_sample_count = 0
            for i in range(n_d):
                if ignore_index != label[i]:
                    unignored_sample_count += 1

            if reduction == "mean" and unignored_sample_count != 0:
                dy_scaled = dy / unignored_sample_count

            for i in range(n_d):
                label_sample = label[i]
                for j in range(c):
                    index = i * c + j
                    if ignore_index == label_sample:
                        d_logit[index] = 0
                    else:
                        d_logit[index] = (
                            np.exp(log_prob[index]) - (label_sample == j) * dy_scaled
                        )

    if len(probability_shape) > 2:
        logit_shape = new_shape
        new_shape = []
        permutations = []
        get_permuation_and_shape(False, logit_shape, new_shape, permutations)
        transpose_output = np.reshape(transpose_output, d_logit.shape)
        d_logit = np.reshape(d_logit, logit_shape)
        transpose_output = np.transpose(d_logit, permutations)
        d_logit = np.reshape(d_logit, new_shape)

    assert d_logit.shape == probability_shape
    return d_logit


# NOTE: This is the DistIR version of Split
# TODO: Merge split and split_v2
def split(op, x):
    dim = op.attributes["axis"]
    if op.op_type == "Split" or op.op_type == "SplitDistIR":
        num_splits = op.attributes["num_splits"]
    elif op.op_type == "MPIScatter" or op.op_type == "MPIScatterToTupleType":
        num_splits = len(op.attributes["devices"])
    else:
        raise NotImplementedError(op.op_type)

    try:
        return tuple(y for y in np.split(x, num_splits, axis=dim))
    except Exception as e:
        import pdb

        pdb.set_trace()


# NOTE: This is the ONNX version of Split
def split_v2(op, x):
    split = op.attributes["split"]
    sections = []
    n = 0
    for s in split[:-1]:
        sections.append(n + s)
        n += s
    axis = op.attributes["axis"]
    return np.split(x, sections, axis=axis)


def sub(op, x, y):
    return x - y


def sum_(op, *xs):
    return sum(xs)


def tanh(op, x):
    return np.tanh(x)


def transpose(op, x):
    perm = op.attributes["perm"]
    return np.transpose(x, perm)


def unsqueeze(op, x):
    axes = op.attributes["axes"]
    for i in axes[::-1]:
        x = np.expand_dims(x, axis=i)
    return x


NumPyRegister = {
    ("Add", (np.ndarray, np.ndarray)): add,
    ("Add", (np.ndarray, np.float32)): add,
    (
        "BiasFastGeluGrad_dX",
        (np.ndarray, np.ndarray, np.ndarray),
    ): bias_fast_gelu_grad_dx,
    ("Cast", (np.ndarray,)): cast,
    ("Cast", (np.int64,)): cast,
    ("Cast", (np.float64,)): cast,
    ("Concat", (tuple,)): concat,
    ("Concat", (np.ndarray, np.ndarray)): concat,
    ("Concat", (np.ndarray, np.ndarray, np.ndarray)): concat,
    ("Concat", (np.ndarray, np.ndarray, np.ndarray, np.ndarray)): concat,
    ("Concat", (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)): concat,
    ("Constant", ()): constant,
    ("ConstantOfShape", (np.ndarray,)): constant_of_shape,
    ("Div", (np.ndarray, np.ndarray)): div,
    ("Div", (np.ndarray, np.float32)): div,
    ("Div", (np.int64, np.int64)): div,
    ("Dropout", (np.ndarray, np.ndarray, bool)): dropout,
    ("DropoutGrad", (np.ndarray, np.ndarray, np.ndarray, np.ndarray)): dropout_grad,
    ("Expand", (np.ndarray, np.ndarray)): expand,
    ("Gather", (np.ndarray, np.ndarray)): gather,
    ("Gather", (np.ndarray, np.int64)): gather,
    ("GatherND", (np.ndarray, np.ndarray)): gather_nd,
    ("GatherNDGrad", (np.ndarray, np.ndarray, np.ndarray)): gather_nd_grad,
    ("GatherGrad", (np.ndarray, np.ndarray, np.ndarray)): gather_grad,
    ("Gemm", (np.ndarray, np.ndarray, np.ndarray)): gemm,
    ("FastGelu", (np.ndarray, np.ndarray)): fast_gelu,
    ("Identity", (np.ndarray,)): identity,
    ("Join", (np.ndarray, np.ndarray)): join,
    ("Join", (np.ndarray, np.ndarray, np.ndarray, np.ndarray)): join,
    ("LayerNormalization", (np.ndarray, np.ndarray, np.ndarray)): layer_norm,
    (
        "LayerNormalizationGrad",
        (np.ndarray, np.ndarray, np.ndarray, np.float64, np.float64),
    ): layer_norm_grad,
    (
        "LayerNormalizationGrad",
        (np.ndarray, np.ndarray, np.ndarray, np.float32, np.float32),
    ): layer_norm_grad,
    ("Loss", (np.ndarray, np.ndarray)): loss,
    ("LossGrad", (np.ndarray, np.ndarray)): loss_grad,
    ("MatMul", (np.ndarray, np.ndarray)): matmul,
    ("MatMulGrad", (np.ndarray, np.ndarray, np.ndarray)): matmul_grad,
    ("Min", (np.ndarray, np.ndarray)): lambda op, x, y: np.minimum(x, y),
    (
        "MPIAllreduceFromTupleType",
        (tuple,),
    ): lambda op, *xs: mpi_allreduce(op, *xs[0]),
    ("MPIAllgather", (np.ndarray,) * 2): mpi_allgather,
    ("MPIAllgather", (np.ndarray,) * 4): mpi_allgather,
    ("MPIAllgather", (np.ndarray,) * 8): mpi_allgather,
    ("MPIAllgather", (np.ndarray,) * 16): mpi_allgather,
    ("MPIAllgather", (np.ndarray,) * 32): mpi_allgather,
    ("MPIAllgather", (np.ndarray,) * 64): mpi_allgather,
    ("MPIAllgather", (np.ndarray,) * 128): mpi_allgather,
    ("MPIAllgather", (np.ndarray,) * 256): mpi_allgather,
    ("MPIAllgather", (np.ndarray,) * 512): mpi_allgather,
    ("MPIAllgather", (np.ndarray,) * 1024): mpi_allgather,
    ("MPIAllgather", (np.ndarray,) * 2048): mpi_allgather,
    ("MPIAllgather", (np.ndarray,) * 4096): mpi_allgather,
    ("MPIAllgather", (np.ndarray,) * 8192): mpi_allgather,
    ("MPIAllreduce", (np.ndarray,) * 2): mpi_allreduce,
    ("MPIAllreduce", (np.ndarray,) * 4): mpi_allreduce,
    ("MPIAllreduce", (np.ndarray,) * 8): mpi_allreduce,
    ("MPIAllreduce", (np.ndarray,) * 16): mpi_allreduce,
    ("MPIAllreduce", (np.ndarray,) * 32): mpi_allreduce,
    ("MPIAllreduce", (np.ndarray,) * 64): mpi_allreduce,
    ("MPIAllreduce", (np.ndarray,) * 128): mpi_allreduce,
    ("MPIAllreduce", (np.ndarray,) * 256): mpi_allreduce,
    ("MPIAllreduce", (np.ndarray,) * 512): mpi_allreduce,
    ("MPIAllreduce", (np.ndarray,) * 1024): mpi_allreduce,
    ("MPIAllreduce", (np.ndarray,) * 2048): mpi_allreduce,
    ("MPIAllreduce", (np.ndarray,) * 4096): mpi_allreduce,
    ("MPIAllreduce", (np.ndarray,) * 8192): mpi_allreduce,
    ("MPIBroadcast", (np.ndarray,)): mpi_broadcast,
    ("MPIBroadcastToTupleType", (np.ndarray,)): mpi_broadcast,
    ("MPIGather", (np.ndarray,) * 2): mpi_gather,
    ("MPIGather", (np.ndarray,) * 4): mpi_gather,
    ("MPIGather", (np.ndarray,) * 8): mpi_gather,
    ("MPIGather", (np.ndarray,) * 16): mpi_gather,
    ("MPIGather", (np.ndarray,) * 32): mpi_gather,
    ("MPIGather", (np.ndarray,) * 64): mpi_gather,
    ("MPIGather", (np.ndarray,) * 128): mpi_gather,
    ("MPIGather", (np.ndarray,) * 256): mpi_gather,
    ("MPIGather", (np.ndarray,) * 512): mpi_gather,
    ("MPIGather", (np.ndarray,) * 1024): mpi_gather,
    ("MPIGather", (np.ndarray,) * 2048): mpi_gather,
    ("MPIGather", (np.ndarray,) * 4096): mpi_gather,
    ("MPIGather", (np.ndarray,) * 8192): mpi_gather,
    ("MPIGatherFromTupleType", (tuple,)): lambda op, *xs: mpi_gather(op, *xs[0]),
    ("MPIReduce", (np.ndarray,) * 2): mpi_reduce,
    ("MPIReduce", (np.ndarray,) * 4): mpi_reduce,
    ("MPIReduce", (np.ndarray,) * 8): mpi_reduce,
    ("MPIReduce", (np.ndarray,) * 16): mpi_reduce,
    ("MPIReduce", (np.ndarray,) * 32): mpi_reduce,
    ("MPIReduce", (np.ndarray,) * 64): mpi_reduce,
    ("MPIReduce", (np.ndarray,) * 128): mpi_reduce,
    ("MPIReduce", (np.ndarray,) * 256): mpi_reduce,
    ("MPIReduce", (np.ndarray,) * 512): mpi_reduce,
    ("MPIReduce", (np.ndarray,) * 1024): mpi_reduce,
    ("MPIReduce", (np.ndarray,) * 2048): mpi_reduce,
    ("MPIReduce", (np.ndarray,) * 4096): mpi_reduce,
    ("MPIReduce", (np.ndarray,) * 8192): mpi_reduce,
    ("MPIScatter", (np.ndarray,)): split,
    ("MPIScatterToTupleType", (np.ndarray,)): split,
    ("Mul", (np.ndarray, np.ndarray)): mul,
    ("Mul", (np.ndarray, np.float32)): mul,
    ("Mul", (np.int64, np.int64)): mul,
    ("NonZero", (np.ndarray,)): lambda op, x: np.array(np.nonzero(x)),
    ("Pow", (np.ndarray, np.float32)): lambda op, x, y: pow(x, y),
    ("ReduceAllL2", tuple(np.ndarray for i in range(60))): reduce_all_l2,
    ("ReduceAllL2", tuple(np.ndarray for i in range(61))): reduce_all_l2,
    ("ReduceAllL2", tuple(np.ndarray for i in range(62))): reduce_all_l2,
    ("ReduceAllL2", tuple(np.ndarray for i in range(63))): reduce_all_l2,
    ("ReduceAllL2", tuple(np.ndarray for i in range(64))): reduce_all_l2,
    ("ReduceMean", (np.ndarray,)): reduce_mean,
    ("ReduceSum", (np.ndarray,)): reduce_sum,
    ("Relu", (np.ndarray,)): relu,
    ("ReluGrad", (np.ndarray, np.ndarray)): relu_grad,
    ("Reshape", (np.ndarray, np.ndarray)): reshape,
    ("Select", (tuple,)): select,
    ("Send", (np.int64,)): identity,
    ("Send", (np.ndarray,)): identity,
    ("Shape", (np.ndarray,)): shape,
    ("Slice", (np.ndarray, np.ndarray, np.ndarray, np.ndarray)): slice_conc,
    ("Slice", (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.int64)): slice_conc,
    ("SplitDistIR", (np.ndarray,)): split,
    ("Split", (np.ndarray,)): split_v2,
    ("Softmax", (np.ndarray,)): softmax,
    ("SoftmaxCrossEntropyLoss", (np.ndarray, np.ndarray)): softmax_cross_entropy_loss,
    (
        "SoftmaxCrossEntropyLossGrad",
        (np.ndarray, np.ndarray, np.ndarray),
    ): softmax_cross_entropy_loss_grad,
    (
        "SoftmaxCrossEntropyLossGrad",
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray),
    ): softmax_cross_entropy_loss_grad,
    ("SoftmaxGrad", (np.ndarray, np.ndarray)): softmax_grad,
    ("Sqrt", (np.ndarray,)): lambda op, x: np.sqrt(x),
    ("Squeeze", (np.ndarray,)): lambda op, x: np.squeeze(x),
    ("Sub", (np.ndarray, np.ndarray)): sub,
    ("Sub", (np.int64, np.int64)): sub,
    ("Sub", (np.float32, np.ndarray)): sub,
    ("Sum", (np.ndarray, np.ndarray)): sum_,
    ("Sum", (np.ndarray, np.ndarray, np.ndarray, np.ndarray)): sum_,
    ("Tanh", (np.ndarray,)): tanh,
    ("Transpose", (np.ndarray,)): transpose,
    ("Unsqueeze", (np.int64,)): unsqueeze,
    ("Unsqueeze", (np.ndarray,)): unsqueeze,
}
