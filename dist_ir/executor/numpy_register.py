import numpy as np


def add(op, inputs):
    return np.add(inputs[0], inputs[1])


def allreduce(op, inputs):
    # TODO: Add attribute for reduction operator
    sum_ = np.sum(inputs[0], axis=0)
    return [sum_ for i in range(len(inputs[0]))]


def bias_fast_gelu_grad_dx(op, inputs):
    kAlpha = np.sqrt(np.pi) * np.sqrt(0.5)
    kGamma = 0.044715
    kBeta = kGamma * kAlpha * 3.0
    dy, x, b = inputs
    input_shape = x.shape
    bias_shape = b.shape
    x_cube = np.power(x, 3)
    tanh_result = kAlpha * np.tanh(x + kGamma * x_cube)
    sech_sqr_result = 1 - (tanh_result * tanh_result)
    dx = dy * (
        0.5 * (tanh_result + sech_sqr_result * (kAlpha * x + kBeta + x_cube) + 1)
    )
    return dx


def broadcast(op, inputs):
    return [inputs[0] for _ in range(len(op.attributes["devices"]))]


def cast(op, inputs):
    proto_dtype = op.attributes["to"]
    if proto_dtype == 0:
        raise ValueError("Undefined data type")
    elif proto_dtype == 1:
        return inputs[0].astype(np.float32)
    elif proto_dtype == 6:
        return inputs[0].astype(np.int32)
    elif proto_dtype == 7:
        return inputs[0].astype(np.int64)
    elif proto_dtype == 9:
        return inputs[0].as_type(np.bool_)
    else:
        raise NotImplementedError(f"Unsupported data type {proto_dtype}")


def concat(op, inputs):
    dim = op.attributes["dim"]
    return np.concatenate(inputs, axis=dim)


def div(op, inputs):
    return inputs[0] / inputs[1]


def dropout(op, inputs):
    x, ratio, training_mode = inputs
    if training_mode:
        scale = 1.0 / (1.0 - ratio)
        mask = np.random.randint(0, 2, size=x.shape)
        x = scale * mask * x
        assert x.shape == inputs[0].shape
        return x, mask
    else:
        return x


def dropout_grad(op, inputs):
    # TODO: Handle 4th input?
    dy, mask, ratio, _ = inputs

    if ratio == 0:
        return dy
    else:
        return mask * dy / (1.0 - ratio)


def expand(op, inputs):
    return inputs[0] * np.ones(inputs[1])


def fast_gelu(op, inputs):
    # https://github.com/hendrycks/GELUs
    x = inputs[0]
    return 1.0 / (1.0 + np.exp(-1.702 * x))


def gather(op, inputs):
    axis = op.attributes["axis"]
    return np.take(inputs[0], inputs[1].astype(np.int64), axis=axis)


def gather_grad(op, inputs):
    # TODO: implement
    return np.zeros(inputs[0])


def gather_nd(op, inputs):
    data, indices = inputs
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


def gather_nd_grad(op, inputs):
    input_shape, indices_tensor, update_tensor = inputs
    batch_dims = op.attributes["batch_dims"]
    indices_shape = indices_tensor.shape
    last_indices_dimension = batch_dims + indices_shape[-1]
    assert last_indices_dimension <= len(input_shape)
    output_tensor = np.zeros(input_shape)

    # TODO: Finish implementing
    """
    grad_size = update_tensor.ndim
    slice_size = element_count_per_slice
    for i in range(grad_size):
        slice_offset = slice_offsets[i // slice_size]
        j = i % slice_size
        output_tensor[slice_offset + j] += update_tensor[i]
    """
    return output_tensor


def mpi_gather(op, inputs):
    dim = op.attributes["dim"]
    return np.concatenate(inputs[0], axis=dim)


def gemm(op, inputs):
    alpha = op.attributes["alpha"]
    beta = op.attributes["beta"]
    transA = op.attributes["transA"]
    transB = op.attributes["transB"]
    a, b, c = inputs
    if transA:
        a = a.T
    if transB:
        b = b.T
    return np.matmul(alpha * a, beta * b) + c


def identity(op, inputs):
    return inputs[0]


def layer_norm(op, inputs):
    eps = 1e-5
    x, scale, beta = inputs
    mean = np.mean(x)
    std = np.std(x)
    x = (x - mean) / (pow(std, 2) + eps) * scale + beta
    assert x.shape == inputs[0].shape
    return x, mean, std


def layer_norm_grad(op, inputs):
    y_grad, x, scale, mean, inv_std_var = inputs
    assert mean == np.mean(x)
    a = y_grad * (x - mean) * inv_std_var
    b = y_grad * scale * inv_std_var
    c = y_grad * scale * inv_std_var * (x - mean) * inv_std_var
    x_grad = b - (x - mean) * inv_std_var * np.mean(c, axis=0)
    bias_grad = np.sum(y_grad, axis=1)
    scale_grad = np.sum(a, axis=1)
    return x_grad, bias_grad, scale_grad


def loss(op, inputs):
    N = op.attributes["N"]
    return np.square(inputs[0] - inputs[1]) / N


def loss_grad(op, inputs):
    N = op.attributes["N"]
    return 2 * (inputs[0] - inputs[1]) / N


def matmul(op, inputs):
    return np.matmul(inputs[0], inputs[1])


def matmul_grad(op, inputs):
    return (np.dot(inputs[2], inputs[1].T), np.dot(inputs[0].T, inputs[2]))


def min_(op, inputs):
    return np.minimum(*inputs)


def mul(op, inputs):
    return inputs[0] * inputs[1]


def relu(op, inputs):
    return np.maximum(inputs[0], 0)


def reshape(op, inputs):
    new_shape = np.copy(inputs[1])
    for i in range(len(new_shape)):
        if new_shape[i] == 0:
            new_shape[i] = inputs[0].shape[i]
    return np.reshape(inputs[0], new_shape)


def select(op, inputs):
    dim = op.attributes["dim"]
    return inputs[0][dim]


def shape(op, inputs):
    return np.array(inputs[0].shape)


def slice_(op, inputs):
    x, starts, ends, axes = inputs
    slices = {axis: slice(s, e) for (s, e, axis) in zip(starts, ends, axes)}
    slices = tuple(slices.get(d, slice(None)) for d in range(x.ndim))
    return x[slices]


def softmax(op, inputs):
    axis = op.attributes["axis"]
    exp = np.exp(inputs[0])
    return exp / np.sum(exp, axis=axis, keepdims=True)


def softmax_grad(op, inputs):
    raise NotImplementedError("softmax_grad")


def softmax_cross_entropy_loss(op, inputs):
    x, target = inputs
    weight = None
    if hasattr(op.attributes, "ignore_index"):
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


def softmax_cross_entropy_loss_grad(op, inputs):
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

    dy, log_prob, label = inputs[:3]
    if hasattr(op.attributes, "ignore_index"):
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

    if len(inputs) == 4:
        weight = inputs[3]
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


def split(op, inputs):
    dim = op.attributes["dim"]
    if op.op_type == "Split":
        num_splits = op.attributes["num_splits"]
    elif op.op_type == "Scatter":
        num_splits = len(op.attributes["devices"])

    return np.split(inputs[0], num_splits, axis=dim)


def sub(op, inputs):
    return inputs[0] - inputs[1]


def sum_(op, inputs):
    return sum(*inputs)


def tanh(op, inputs):
    return np.tanh(inputs[0])


def transpose(op, inputs):
    perm = op.attributes["perm"]
    return np.transpose(inputs[0], perm)


def unsqueeze(op, inputs):
    x = inputs[0]
    axes = op.attributes["axes"]
    # TODO: Does this need to be in reverse order?
    for i in axes:
        x = np.expand_dims(x, axis=i)
    return x


NumPyRegister = {
    "Add": add,
    "Allreduce": allreduce,
    "BiasFastGeluGrad_dX": bias_fast_gelu_grad_dx,
    "Broadcast": broadcast,
    "Cast": cast,
    "Concat": concat,
    "Div": div,
    "Dropout": dropout,
    "DropoutGrad": dropout_grad,
    "Expand": expand,
    "FastGelu": fast_gelu,
    "Gather": gather,
    "GatherGrad": gather_grad,
    "GatherND": gather_nd,
    "GatherNDGrad": gather_nd_grad,
    "Gemm": gemm,
    "Identity": identity,
    "LayerNormalization": layer_norm,
    "LayerNormalizationGrad": layer_norm_grad,
    "Loss": loss,
    "LossGrad": loss_grad,
    "MatMul": matmul,
    "MatMulGrad": matmul_grad,
    "Min": min_,
    "MPIGather": mpi_gather,
    "Mul": mul,
    "Relu": relu,
    "Reshape": reshape,
    "Scatter": split,
    "Select": select,
    "Send": identity,
    "Shape": shape,
    "Slice": slice_,
    "Softmax": softmax,
    "SoftmaxGrad": softmax_grad,
    "SoftmaxCrossEntropyLoss": softmax_cross_entropy_loss,
    "SoftmaxCrossEntropyLossGrad": softmax_cross_entropy_loss_grad,
    "Split": split,
    "Sub": sub,
    "Sum": sum_,
    "Tanh": tanh,
    "Transpose": transpose,
    "Unsqueeze": unsqueeze,
}
