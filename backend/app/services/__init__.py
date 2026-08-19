"""Service layer: orchestration across adapters and domain functions.

Services own sequencing and consistency. They hold no parsing, chunking or
similarity logic of their own -- that belongs to ``app.domain`` and
``app.adapters`` -- and they never import FastAPI.
"""
