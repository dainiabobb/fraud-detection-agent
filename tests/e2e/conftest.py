"""E2E test configuration — registers custom CLI options."""


def pytest_addoption(parser):
    parser.addoption(
        "--sample-size",
        default=1000,
        type=int,
        help="Number of IEEE-CIS transactions to sample (default: 1000).",
    )
