from dist_ir.executor import (
    calibrate_device_parameters,
    calibrate_network_bandwidth,
    network_bandwidth_debug,
)


def main():
    """
    (
        dram_bandwidth,
        device_throughput,
        kernel_launch_overhead,
    ) = calibrate_device_parameters()
    print(f"Device throughput: {device_throughput:e}")
    print(f"DRAM bandwidth: {dram_bandwidth:.2e}")
    print(f"Kernel launch overhead: {kernel_launch_overhead}")
    network_bandwidth = calibrate_network_bandwidth()
    print(f"Network bandwidth: {network_bandwidth}")
    """
    bandwidths = calibrate_network_bandwidth()
    for k, v in bandwidths.items():
        print(f"{k}: {v}")
    # network_bandwidth_debug()


if __name__ == "__main__":
    main()
