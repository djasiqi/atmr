"""Package services — point d'entrée pour imports et patches de tests.

Sans ``__init__.py``, certains ``patch("services.<sous_module>...")`` échouent
car le parent ``services`` n'expose pas les sous-modules comme attributs.
"""

from __future__ import annotations

__all__: list[str] = []
