"""Text normalization for rail MATCHING only (never for stored/shown text).

Shared by ``input_checks`` and ``output_checks`` so obfuscation handling is
identical on both rails:
  1. NFKC — full-width / compatibility forms → ASCII (full-width ``has`` → ``has``),
     NBSP → space;
  2. strip zero-width characters (ZWSP/ZWNJ/ZWJ/WJ/BOM) and soft hyphens;
  3. fold common Cyrillic/Greek Latin-lookalike letters to Latin
     (``h<U+0430>s`` with Cyrillic U+0430 → ``has``);
  4. collapse whitespace runs to a single space.
"""

import re
import unicodedata

_INVISIBLE = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
_WHITESPACE = re.compile(r"\s+")

# Latin lookalikes (lower + upper) from Cyrillic and Greek. Deliberately small:
# letters that render identically to ASCII in common fonts.
_CONFUSABLES = str.maketrans(
    (
        "\u0430\u0435\u043e\u0440\u0441\u0443\u0445\u0456\u0458\u0455\u04bb\u0501"
        "\u0261\u0410\u0412\u0415\u041a\u041c\u041d\u041e\u0420\u0421\u0422\u0425"
        "\u0406\u0408\u0405\u03b1\u03bf\u03c1\u03bd\u03b9\u03ba\u03c4\u0391\u0392"
        "\u0395\u0397\u0399\u039a\u039c\u039d\u039f\u03a1\u03a4\u03a7\u03a5\u0396"
    ),
    "aeopcyxijshdgABEKMHOPCTXIJSaopviktABEHIKMNOPTXYZ",
)


def normalize_for_matching(text: str) -> str:
    folded = _INVISIBLE.sub("", unicodedata.normalize("NFKC", text)).translate(_CONFUSABLES)
    return _WHITESPACE.sub(" ", folded)
