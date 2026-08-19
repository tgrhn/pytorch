### 🚀 The feature, motivation and pitch

# RFC: Sweep and Refactor Device-Coupled Code Toward a Device-Agnostic System

## Summary

This RFC proposes an active initiative: search the PyTorch codebase for code that unnecessarily hardcodes assumptions about a single accelerator backend (CUDA, XPU, MTIA, MPS, or any other), and refactor those spots to use PyTorch's existing generic accelerator APIs (`torch.accelerator.current_accelerator()`, generic per-backend module lookup, etc.) instead. The end state is a codebase where a given piece of logic works the same way regardless of which accelerator is actually present, except where the underlying behavior is genuinely backend-specific (a vendor library, a hardware-capability-gated kernel) and has no cross-backend equivalent to generalize to.

## Motivation

### Example

`torch/_utils.py` contains `_get_available_device_type()`, which used to look like this:

```python
def _get_available_device_type():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if hasattr(torch, "mtia") and torch.mtia.is_available():
        return "mtia"
    custom_backend_name = torch._C._get_privateuse1_backend_name()
    custom_device_mod = getattr(torch, custom_backend_name, None)
    if custom_device_mod and custom_device_mod.is_available():
        return custom_backend_name
    # add more available device types here
    return None
```

and a sibling function, `_get_device_attr()`, manually re-dispatched on the result with another hardcoded if/elif chain to decide which `torch.<backend>` module to call into. Every time a new accelerator was added to PyTorch, someone had to remember to come back to this exact function and add another `if` branch — the comment `# add more available device types here` is a literal invitation to keep doing that by hand, forever.

PyTorch already has a generic primitive that does this detection once, centrally, in C++ (`torch.accelerator.current_accelerator()`), and a generic per-backend module lookup pattern (`getattr(torch, device_type)`) already used elsewhere in the same file. Refactoring `_get_available_device_type()`/`_get_device_attr()` to use these reduced ~20 lines of manually maintained per-backend branching to:

```python
def _get_available_device_type():
    if (acc := torch.accelerator.current_accelerator(check_available=True)) is not None:
        return acc.type
    return None

def _get_device_attr(get_member):
    device_type = _get_available_device_type()
    if not device_type:
        return None
    return get_member(_get_device_module(device_type))
```

with identical behavior for every backend that worked before, no per-backend branch to maintain, and no future edit needed when the next accelerator is added. This is the shape of fix this initiative is looking for: a real, hand-maintained per-backend chain, replaced by an existing generic primitive, verified to preserve behavior.

### Why this needs active work, not just awareness

This kind of coupling doesn't get cleaned up on its own:

* It's added incrementally — each backend was added correctly at the time by extending an existing chain, and no single change looks wrong in isolation.
* It's invisible unless someone is specifically looking for it — the code works fine on whichever backend the original author tested on.
* Nobody owns "go find and fix these" as an ongoing job by default, so they accumulate until someone runs a dedicated sweep.

## Goals

1. Search the codebase for hardcoded single-backend chains, checks, and dispatch logic that duplicate what a generic accelerator API already provides.
2. Refactor confirmed instances to use the generic API, verifying behavior is preserved (including edge cases like priority order when multiple accelerators could be present, and no-accelerator fallback behavior).
3. Land these as scoped, individually reviewable fixes — not one giant sweep-wide commit — each with its own verification.
4. Leave genuinely backend-specific code untouched (vendor libraries, hardware-capability-gated kernels, backend-specific modules). This initiative fixes real coupling, it doesn't attempt to eliminate all backend-specific code.

## Non-Goals

* This does not mean rewriting inherently vendor-locked subsystems (cuDNN bindings, vendor-specific attention kernels, vendor-library-backed ops) to be backend-generic. Those are supposed to reference their vendor directly.
* This is not a mechanical find-and-replace of every occurrence of a backend's name. A hardcoded check that gates access to something only that backend actually implements is correct and should stay as-is — replacing it with a generic accelerator call would let other backends reach code that was never built to support them, which is worse than the original hardcoding.
* This does not include an automated lint rule in this pass. That could be a reasonable follow-up once enough real fixes establish a precise, low-noise pattern to detect mechanically.

## What to look for while sweeping

Concretely, when scanning a file:

* A hardcoded if/elif chain detecting "which accelerator is available" or "which backend module to use" — especially with a comment inviting future manual additions per backend. Candidate for `torch.accelerator.current_accelerator()`.
* A function that reimplements generic per-backend module dispatch (`if device == "cuda": mod = torch.cuda` / `elif device == "xpu": mod = torch.xpu` / ...) instead of `getattr(torch, device_type)`.
* Before applying a fix, confirm the code isn't actually gating access to something implemented for only one backend (a vendor kernel, a hardware-capability check) — if it is, leave it as-is.
* When replacing a chain with `torch.accelerator.current_accelerator()`, check whether the priority order matches the old chain's (which backend wins if more than one is technically present) — if it differs, call that out explicitly as a behavior change, not a pure refactor.
* When a fix uses `getattr(torch, device_type)` to reach a sub-attribute (e.g. `<module>.memory.<function>`), verify that attribute actually exists on every backend module being targeted — backend modules aren't guaranteed to have the same shape.

## Where to Start

A few concrete searches to surface candidates:

```bash
# Hardcoded per-backend if/elif chains (the main pattern this initiative targets)
grep -rn 'if.*torch\.cuda\.is_available()' --include=*.py torch/
grep -rn 'if hasattr(torch, "xpu")' --include=*.py torch/

# Comments that flag exactly this kind of manually-maintained list
grep -rn 'add more.*device\|add more.*backend\|add more.*accelerator' --include=*.py torch/

# Manual per-backend module dispatch that could be getattr(torch, device_type)
grep -rn 'if device.*==.*"cuda".*:\s*$' --include=*.py torch/ -A 1
```

These are starting points, not an exhaustive list — treat each hit through the "what to look for" checks above before touching anything, since most will turn out to be correct as-is.

## Conclusion

This RFC proposes doing the actual refactoring work — finding hardcoded single-backend chains and dispatch logic across the codebase and replacing them with PyTorch's existing generic accelerator primitives, one verified fix at a time — rather than producing a policy document about how such reviews should be conducted in the abstract.

### Alternatives

_No response_

### Additional context

_No response_
