# Source attribution

The inference runtime is derived from MR0 / StarVLA (Qwen3.5 + MMDiT).
StarVLA copyright and its supplied license are retained in LICENSE and in file
headers. Public upstream contributor credits are retained. The policy is not
claimed as a new model created by the packaging author.

The simulation evaluation, runtime contracts and observation/video wrappers
derive from the RoboCasa benchmark's Isaac-GR00T evaluation integration,
revision `9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10`:
https://github.com/robocasa-benchmark/Isaac-GR00T
Their NVIDIA copyright headers and Apache-2.0 license are retained in
licenses/NVIDIA-GR00T-LICENSE. Local adaptation provides the Native12/State20
policy bridge and four-observation history support.

The logging utility acknowledges OpenVLA / Prismatic in its original header.
The MMDiT action head acknowledges the Psi0 architecture in its original header.
Qwen, RoboCasa, robosuite and their downloaded assets remain subject to their
own licenses. This repository does not redistribute training demonstrations,
kitchen assets or pretrained weights.

This is a fresh evaluation-only snapshot. Original repository history was not
available in the source snapshot; no original commits are fabricated here.
Private source-control URLs, host paths and author-specific run identifiers
have been omitted. `provenance/code_manifest.json` records relative source
filenames and file hashes. Mathematically relevant preprocessing, normalization,
history handling and action decoding are preserved.
