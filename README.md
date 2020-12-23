# DistIR

An IR to represent distributed computations.
Our main goal is an IR capable of representing the complex distributed strategies
used in large-scale distributed training, while at the same time enabling fast
cost models/simulations.

Distributed strategies we want to support:
- Data parallelism
- Horizontal parallelism
- Pipeline parallelism
- Megatron
- ZeRO partitioning
- PyTorch lightning's Zero
- Stashing & recomputation
- Overlapping computation and communication
- Local layer parallelism

# Requirements/Installation

See the [build file](.github/workflows/tests.yml) and [PIP packages list](requirements.txt).

# Directory structure

- dist_ir: Python source for DistIR
    - ir: IR definitions
    - importer: create DistIR from ONNX/MLIR
    - executor: ways to "execute" or simulate DistIR
- docs: documentation and notes
- notebooks: small experiments and worked examples
- test: unit tests, small/toy example models

# Running tests

Run the following from the root of this repository:
```bash
python -m pytest
```

# Components

- Executors:
    - SequentialExecutor: a reference implementation that runs a DistIR module
        on a single device. Can be used to check correctness of transforms.
    - DistributedSimulator: an executor that uses profile data or flop counts to
        simulate the execution of a given DistIR module on a given hardware
        configuration (including communication bandwidths and processor speed).
        Returns estimated execution time and live memory profile. This can be
        split into three subcomponents:
        - Shape Inference: a pass that uses the shapes of inputs to calculate
            the shapes of all intermediate values.
        - Cost Inference: a pass that uses either shape information to compute
            (or profiles the module and measures) the runtime and temporary
            memory requirement of each op in the module.
            This output can be cached.
        - Simulator: takes a module and a mapping from op to time/memory
            consumption and does a simulation to obtain a concurrent trace
            (from which total runtime and memory usage plots can be derived).
- Importers:
    - ONNX Importer: convert a `.onnx` file to a DistIR module. Can be given an
        intermediate graph from ORT (for example, after AD).
    - MLIR Importer: import a DistIR module written in MLIR text format to an
        in-memory DistIR module object. TODO
- Exporter/Prettyprinter: converts a DistIR module to an MLIR text format string.
- Transforms: a module containing DistIR->DistIR transforms.
    Ideally, these modules should be composable and should run on submodules
    so that we can have nested parallelism (data parallel where a subset of the
    layers are horizontal parallel, or pipeline parallel where each stage is
    data parallel with a different degree).
    - DataParallelTransform: converts a given DistIR module to a data-parallel
        version that runs on a given number of devices.
    - HorizontalParallelTransform: converts a given DistIR module to a
        horizontal-parallel version (if possible) that runs on a given number of
        devices.
    - PipelineParallelTransform: converts a given DistIR module to a
        pipeline-parallel version that runs on a given number of devices.
- Search: an algorithm to find the best distributed version of a given
    sequential DistIR module. Initially, this can be something that searches
    the DHP space (i.e. find the optimal parameters to give the D/H/P transforms).
