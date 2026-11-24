"""Translate module names in PyTorch checkpoints without copying weight tensors."""

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy


def rename_state_dict_prefix(state_dict, old_name, new_name):
    """Rename a module and its metadata, using an empty name for the model root."""
    if not isinstance(state_dict, Mapping):
        raise TypeError("Expected a state_dict mapping")
    old_prefix = old_name + "." if old_name else ""
    new_prefix = new_name + "." if new_name else ""

    def rename_keys(mapping, *, metadata=False):
        result = OrderedDict()
        promote_root = metadata and bool(old_name) and not new_name and old_name in mapping
        for key, value in mapping.items():
            if key == "" and promote_root:
                # Unwrapping promotes the model's metadata over its wrapper's.
                continue
            if metadata and key == old_name:
                renamed = new_name
            elif key.startswith(old_prefix):
                renamed = new_prefix + key[len(old_prefix):]
            else:
                renamed = key
            if renamed in result:
                raise ValueError(
                    f"Duplicate checkpoint key {renamed!r} after renaming "
                    f"{old_name!r} to {new_name!r}"
                )
            result[renamed] = value
        return result

    renamed = rename_keys(state_dict)
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        renamed._metadata = rename_keys(deepcopy(metadata), metadata=True)
    return renamed
