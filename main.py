import logging

from electwit.sim import ElecTwit

logger = logging.getLogger(__name__)

from electwit.tests.test_sim import test_simulation


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    test_simulation()
    logger.info("Simulation test completed successfully.")
