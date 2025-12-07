def initialize_numpyro(num_cpu_devices: int = 4):  # For 4 parallel chains
    import os

    import numpyro  # type: ignore[import-untyped]

    # To avoid file lock error: https://github.com/pymc-devs/pymc/issues/6818
    os.environ["PYTENSOR_FLAGS"] = (
        f"compiledir_format=compiledir_{os.getpid()},base_compiledir={os.path.expanduser('~')}/.pytensor/compiledir_llm-mcts"  # Avoid file lock error
    )
    numpyro.set_platform("cpu")  # Use CPU rather than GPU for sample_numpyro_nuts
    numpyro.set_host_device_count(
        max(1, num_cpu_devices)
    )  # https://github.com/CDCgov/PyRenew/issues/151#issuecomment-2386861351
