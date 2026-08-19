"""Adapters: concrete implementations of the ports the services depend on.

Every adapter sits behind a Protocol declared in its own ``base`` module, so a
service is written against the interface and never against a vendor SDK. No
adapter imports FastAPI.
"""
