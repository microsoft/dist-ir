# DistIR

An IR to represent distributed computations.
Our main goal is an IR capable of representing the complex distributed strategies
used in large-scale distributed training, while at the same time enabling fast
cost models.

Distributed strategies we want to support:
- D
- H
- P
- Stashing & recomputation
- Overlapping computation and communication

# Directory structure

- dist_ir: Python source for DistIR
    - executor: TODO
    - graph: TODO
    - ops: TODO
    - tests: TODO merge with test
- test: unit tests, small/toy example models
- docs: documentation and notes

# Running tests

TODO

# Components

- Executors:
    - SequentialExecutor: a reference implementation that runs a DistIR module
        on a single device. Can be used to check correctness of transforms.
    - DistributedSimulator: an executor that uses profile data or flop counts to
        simulate the execution of a given DistIR module on a given hardware
        configuration (including communication bandwidths and processor speed).
        Returns estimated execution time and live memory profile.
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