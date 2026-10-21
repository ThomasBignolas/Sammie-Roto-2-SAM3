"""Text list syntax and collision-free IDs for independent SAM concept sessions."""
import re

OBJECT_ID_STRIDE = 1_000_000
MAX_OBJECTS = 16


def split_concepts(text):
    """Commas, semicolons and 'and' separate searches; quotes preserve a phrase."""
    chunks = re.split(r'("[^"\n]*")', text or '')
    phrases, current = [], ''
    for chunk in chunks:
        if chunk.startswith('"') and chunk.endswith('"') and len(chunk) >= 2:
            current += chunk[1:-1]
            continue
        if '"' in chunk:
            raise ValueError('Close the double quotes around the object description.')
        pieces = re.split(r'[,;\n]|\band\b', chunk, flags=re.IGNORECASE)
        current += pieces[0]
        for piece in pieces[1:]:
            phrases.append(current.strip())
            current = piece
    phrases.append(current.strip())
    unique = {}
    for phrase in phrases:
        if phrase:
            unique.setdefault(phrase.casefold(), phrase)
    if len(unique) > MAX_OBJECTS:
        raise ValueError('Search at most 16 object descriptions at once.')
    return list(unique.values())


def scoped_id(group, native):
    if not 0 <= int(group) < MAX_OBJECTS or not 0 <= int(native) < OBJECT_ID_STRIDE:
        raise ValueError('SAM object/session ID is outside the supported range.')
    return int(group)*OBJECT_ID_STRIDE + int(native)


def split_id(object_id):
    group, native = divmod(int(object_id), OBJECT_ID_STRIDE)
    scoped_id(group, native)  # validate
    return group, native
