# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    """These are pure-Python unit tests that never initialize an accelerator.

    Skipping the global cleanup avoids tearing down a distributed environment
    that was never set up, and is ~10x faster. See the base fixture in
    tests/conftest.py.
    """
    return False
