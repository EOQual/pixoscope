"""Lecteur, en lecture seule, du format ``.lum`` maison.

Se limite à ce dont un visualisateur a besoin : lire l'en-tête et des
plages de lignes. L'écriture du format (utile pour des pipelines de
traitement) est hors périmètre de pixoscope, qui est un outil de
visualisation.

Format ``.lum``
----------------
En-tête binaire de 12 octets : nombre de colonnes (uint32), nombre de
lignes (uint32), puis un code 4 caractères (ex. ``"16LI"``) indiquant le
type de pixel et l'endianness (voir :data:`LUM_CODE_DICTIONARY`). Les
données brutes suivent immédiatement, une ligne après l'autre, sans
compression.
"""

from __future__ import annotations

from pathlib import Path
from struct import unpack

import numpy as np

#: Association code lum -> (dtype numpy, endianness). Sous-ensemble des
#: codages ``.lum`` existants, limité à ceux effectivement lisibles par
#: ce plugin (lecture seule).
LUM_CODE_DICTIONARY: dict[str, tuple[str, str]] = {
    "80IB": ("uint8", "little"),
    "08BI": ("uint8", "big"),
    "16BI": ("uint16", "big"),
    "32BI": ("uint32", "big"),
    "12BI": ("uint16", "big"),
    "13BI": ("uint16", "big"),
    "14BI": ("uint16", "big"),
    "16BS": ("uint16", "big"),
    "01IB": ("uint16", "little"),
    "FLOA": ("float32", "big"),
    "AOLF": ("float32", "little"),
    "08LI": ("uint8", "little"),
    "16LI": ("uint16", "little"),
    "16LS": ("int16", "little"),
    "32LI": ("uint32", "little"),
    "32LU": ("uint32", "little"),
    "32LS": ("int32", "little"),
    "12LI": ("uint16", "little"),
    "13LI": ("uint16", "little"),
    "14LI": ("uint16", "little"),
    "FLOL": ("float32", "little"),
    "08LU": ("uint8", "little"),
    "16LU": ("uint16", "little"),
    "R4L ": ("float32", "little"),
    "R8L ": ("float64", "little"),
}

_TYPE_SIZE: dict[str, int] = {
    "uint8": 1,
    "uint16": 2,
    "int16": 2,
    "uint32": 4,
    "int32": 4,
    "float32": 4,
    "float64": 8,
}


class LumReader:
    """Lecteur en lecture seule d'un fichier ``.lum``.

    Parameters
    ----------
    path : pathlib.Path
        Chemin du fichier ``.lum``.

    Attributes
    ----------
    n_cols, n_rows : int
        Dimensions de l'image.
    dtype : numpy.dtype
        Type des pixels, déduit du code lum.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        with open(path, "rb") as handle:
            n_cols_bytes = handle.read(4)
            n_rows_bytes = handle.read(4)
            code_bytes = handle.read(4)

        code = code_bytes.decode("utf-8")
        if code not in LUM_CODE_DICTIONARY:
            raise ValueError(f"Code lum inconnu [{code!r}] dans le fichier [{path}]")
        codage, endian = LUM_CODE_DICTIONARY[code]

        struct_endian = "<I" if endian == "little" else ">I"
        (self.n_cols,) = unpack(struct_endian, n_cols_bytes)
        (self.n_rows,) = unpack(struct_endian, n_rows_bytes)

        self._codage = codage
        self._endian = endian
        self.dtype = np.dtype(codage)
        self._row_offset_bytes = self.n_cols * _TYPE_SIZE[codage]
        #: Les données commencent après l'en-tête, complété jusqu'à
        #: occuper une ligne entière (et non seulement les 12 octets
        #: qu'il contient) : ce format historique réserve ainsi la
        #: première ligne du fichier.
        self._data_offset = self._row_offset_bytes

    def read_rows(self, start_row: int, n_rows: int) -> np.ndarray:
        """Lit une plage de lignes contiguës, sans charger tout le fichier.

        C'est la seule opération de fenêtrage possible nativement sur ce
        format (pas de tuilage colonnes) — voir la limite documentée dans
        ``REFERENCE_TECHNIQUE.md``.

        Parameters
        ----------
        start_row : int
            Première ligne à lire (0-based).
        n_rows : int
            Nombre de lignes à lire.

        Returns
        -------
        numpy.ndarray
            Tableau ``(n_rows, n_cols)`` dans le dtype natif du fichier.
        """
        start_row = max(0, start_row)
        n_rows = max(0, min(n_rows, self.n_rows - start_row))
        if n_rows == 0:
            return np.zeros((0, self.n_cols), dtype=self.dtype)

        offset = self._data_offset + start_row * self._row_offset_bytes
        with open(self.path, "rb") as handle:
            handle.seek(offset)
            data = np.fromfile(handle, dtype=self.dtype, count=self.n_cols * n_rows)
        data = data.reshape(n_rows, self.n_cols)
        if self._endian != _native_endianness():
            data = data.byteswap()
        return data

    def read_all(self) -> np.ndarray:
        """Lit l'image entière.

        À réserver aux petits fichiers ou à la construction d'un cache
        de pyramide (voir :mod:`pixoscope.io.pyramid_builder`) — préférer
        :meth:`read_rows` pour un affichage interactif.

        Returns
        -------
        numpy.ndarray
            Tableau ``(n_rows, n_cols)``.
        """
        return self.read_rows(0, self.n_rows)


def _native_endianness() -> str:
    import sys

    return "little" if sys.byteorder == "little" else "big"
