"""Evaluation sets: fixed questions and cases with their right answers, kept in Git (evals/).

    library_set.py   questions for the library: which sources answer each, and how well (coverage)
    talk_set.py      invented personal talks: the progress and notes they should yield, and every
                     word that looks like a name, marked as a person or as a word
    masking.py       what anonymization did to each marked word: no model needed

A set is versioned in its file name (library-v1.json). Once results are reported on a version, it is
not edited: a fix or a new case goes into the next version, so numbers stay comparable.
"""

from .library_set import CaseKind, LibraryCase, LibraryEvalSet, load_library_set, unknown_sources
from .talk_set import Mention, TalkCase, TalkEvalSet, load_talk_set

__all__ = [
    "CaseKind",
    "LibraryCase",
    "LibraryEvalSet",
    "Mention",
    "TalkCase",
    "TalkEvalSet",
    "load_library_set",
    "load_talk_set",
    "unknown_sources",
]
