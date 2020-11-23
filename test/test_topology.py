from dist_ir.ir import Topology


def test_topology():
    topology = Topology()

    a = topology.add_device("gpu")
    assert a.device_id == 0
    assert a.device_type == "gpu"

    b = topology.add_device("gpu")
    assert b.device_id == 1
    assert b.device_type == "gpu"

    topology.set_bandwidth(a, b, 100.0)
    assert topology.get_bandwidth(a, b) == 100.0
    assert topology.get_bandwidth(b, a) == 100.0
