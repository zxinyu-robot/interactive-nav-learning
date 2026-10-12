# Interactive Nav Learning

An interruptible, correctable, and uncertainty-aware learning-based navigation
research repository.

The deployed navigation policy is learned end to end. Classical shortest-path
planning may be used only to generate teacher trajectories; it is not part of
runtime inference.

## Initial scope

- Lightweight 2D continuous navigation environment with deterministic seeds.
- Language-conditioned goals and mid-episode corrections.
- Mixed policy outputs: linear velocity, angular velocity, stop, and ask.
- Behavior cloning baseline.
- DAgger intervention collection and retraining.
- Evaluation for success, SPL, collisions, intervention count, correction
  response latency, and control smoothness.
- Optional ROS 2/ONNX deployment adapters kept outside the core package.

## Engineering constraints

- The core test suite must run without Habitat, Isaac Sim, ROS, or a GPU.
- Heavy simulator integrations are optional extras.
- Training and evaluation must be reproducible from versioned configs.
- Safety termination belongs to the environment and evaluation harness; it
  must not be presented as a learned capability.

The full implementation is being built on a remote development branch.
