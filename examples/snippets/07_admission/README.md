# 07 - Admission policy

When an agent is a directory, an *admission policy* is just a function
that inspects the directory and refuses to load it if a rule is
violated. The same pattern the Kubernetes ecosystem uses for OCI
images applies directly.

This snippet ships a policy with three rules:

1. `bash` tool requires a hook whose `class_name` contains `Permission`.
2. The system prompt must not contain any forbidden phrase.
3. `max_steps` must be `<= 50`.

## Try it

```bash
# The shipped coder has bash but its hooks are not named *Permission*,
# so this admission policy denies it. (Real deployments should rename
# the relevant hook or relax the policy.)
uv run python examples/snippets/07_admission/admit.py examples/coder.cartridge

# The refactorer trivially passes because it has no hooks dir of its
# own - and the policy is purely static, so it does not chase
# `extends:` into the parent. This illustrates a real subtlety:
# admission of an inheriting cartridge either needs to resolve the
# parent or inherit policy decisions.
uv run python examples/snippets/07_admission/admit.py \
    examples/snippets/01_inheritance/refactorer.cartridge
```

## Why this matters

The policy is 60 lines of plain Python. It runs in a CI pipeline
beside lint and tests. The same shape ports to OPA/Rego, signed
attestations, or any policy engine an organisation already runs.
The point is not the rules themselves - it is that *the boundary
makes the policy possible*.
