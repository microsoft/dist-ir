# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import wraps


def singleton(cls):
    """Make a class a Singleton class (only one instance). Adapted from
    https://realpython.com/primer-on-python-decorators/#creating-singletons
    """

    @wraps(cls)
    def wrapper_singleton():
        if not wrapper_singleton.instance:
            wrapper_singleton.instance = cls()
        return wrapper_singleton.instance

    wrapper_singleton.instance = cls()
    return wrapper_singleton  # The value that goes into globals() for the module
    # For a singleton class Foo if we write Foo() we get the unique instance.
