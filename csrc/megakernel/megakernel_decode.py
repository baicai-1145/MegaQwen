"""
Thin compatibility wrapper for megakernel bindings.

Real implementation now lives in `megakernel_decode_impl.py` to keep this
entrypoint small and easier to navigate.
"""

import megakernel_decode_impl as _impl

for _name in dir(_impl):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_impl, _name)

__all__ = [n for n in globals() if not n.startswith("__")]


if __name__ == "__main__":
    import runpy

    runpy.run_module("megakernel_decode_impl", run_name="__main__")
