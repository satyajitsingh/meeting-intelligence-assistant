"""Pure domain layer: models, parsing and chunking.

Nothing in this package performs I/O or imports a web framework. Every function
here is deterministic, which is what makes transcript handling testable without
mocks, network access or fixtures on disk.
"""
