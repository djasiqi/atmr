"""Adaptateurs persistence pour le module Dispatch."""

from .assignment_writer import SqlAlchemyAssignmentWriter

__all__ = [
    "SqlAlchemyAssignmentWriter",
]
