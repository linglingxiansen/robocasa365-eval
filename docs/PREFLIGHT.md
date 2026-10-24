# Rendering preflight

On each new host, or after a driver restart:

1. Inspect `nvidia-smi` and the compute-process list. Select a render GPU that
   is not assigned to active training or evaluation. Do not probe busy devices.
2. Run the launcher with `--dry-run` and verify task count, interpreter paths,
   Native12/State20 dimensions, action horizon and GPU assignments.
3. In the simulation environment, run:

   ```bash
   python -m examples.robocasa365.eval_files.official.preflight.check_render_gpu \
     --gpus 0 --mode auto --n-envs 5 --stress-workers 7 \
     --json-output outputs/preflight/render.json
   ```

   Match `--n-envs` and `--stress-workers` to the selected evaluation YAML.
   For `eval_2gpu.yaml`, use `--stress-workers 1`. The probe first tests one
   vector worker and then the complete concurrent layout. It uses the same
   observation/rendering wrapper stack as evaluation.
4. Both phases must pass, every worker must return finite nonempty RGB
   observations and exit cleanly, and `nvidia-smi` must remain responsive.
   A failed, timed-out or blocked probe is a stop condition. Preserve the
   evidence and resolve the host/driver problem before evaluation. Do not
   terminate unrelated jobs or repeatedly relaunch stuck EGL workers.
5. Start the full run only after the checks pass. Confirm all model servers
   connect and every shard progresses beyond environment reset.

Set GPU visibility with `CUDA_VISIBLE_DEVICES` as the launcher does; do not
set `MUJOCO_EGL_DEVICE_ID`. The preflight validates rendering and does not
replace checkpoint loading or model inference checks. Reading configuration,
aggregating existing results and CPU tests require no rendering probe.
