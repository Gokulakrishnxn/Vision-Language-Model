# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Environment & installation

The project is a Python package installed from the repo root with `setup.py` (package source under `src/`). The README’s installation instructions are the source of truth; this section highlights the key steps.

Create and activate the conda environment, install Habitat-Sim, then install this package and its Python dependencies:

```bash
conda create -n vlm_nav python=3.9 cmake=3.14.0
conda activate vlm_nav
conda install habitat-sim=0.3.1 withbullet headless -c conda-forge -c aihabitat

# From the repo root
pip install -e .
pip install -r requirements.txt
```

VLM backends:
- **Gemini**: `GeminiVLM` reads `GEMINI_API_KEY` from the environment (typically via `.env` loaded by `scripts/main.py`).
- **OpenAI-compatible models**: `OpenAIVLM` reads `OPENAI_API_KEY` and optional `OPENAI_BASE_URL` from the environment.

Datasets and assets (summarized from the README):
- HM3D scenes under `data/scene_datasets/hm3d/…` (download via `habitat_sim.utils.datasets_download` using Matterport API credentials).
- ObjectNav episodes under `data/datasets/objectnav_hm3d_v2/...`.
- GOAT-Bench episodes under `data/datasets/goat_bench/hm3d/v1/...`.
- All are expected under a top-level `data/` directory in this repo; see `README.md` for the detailed directory tree.

## Common commands

All commands below assume the conda env is active and you are in the repo root.

### Install / "build" the package

Install the project as an editable Python package (needed before running any scripts):

```bash
pip install -e .
pip install -r requirements.txt
```

### Run a single ObjectNav trajectory (with GIF output)

Runs `ObjectNavEnv` + `ObjectNavAgent` for one episode (as configured in `config/ObjectNav.yaml`) and saves logs and a GIF under `logs/`:

```bash
python scripts/main.py --config ObjectNav
```

Useful overrides (applied to `env_cfg` in the YAML):

```bash
# More steps, more episodes, and a custom run name
python scripts/main.py \
  --config ObjectNav \
  -ms 40 \
  -ne 100 \
  --name debug_objectnav

# Lower logging frequency (only log every 10 steps)
python scripts/main.py --config ObjectNav -lf 10
```

### Run a single GOAT-Bench trajectory

Uses `GOATEnv` + `GOATAgent` as defined in `config/GOAT.yaml`:

```bash
python scripts/main.py --config GOAT
```

You can use the same CLI overrides (`-ms`, `-ne`, `-lf`, `--name`, etc.) as for ObjectNav; they all map into `env_cfg`.

### Parallel evaluation at scale

There are two ways to run many episodes in parallel:

1. **Using the provided tmux + aggregator wrapper (`parallel.sh`)**

   Edit the configuration block at the top of `parallel.sh`:
   - `NUM_GPU`, `INSTANCES`, `NUM_EPISODES_PER_INSTANCE`, `MAX_STEPS_PER_EPISODE`
   - `TASK` (e.g. `ObjectNav`), `CFG` (corresponding YAML in `config/`), `NAME`
   - `SLEEP_INTERVAL`, `LOG_FREQ`, `PORT`, `VENV_NAME`

   Then run:

   ```bash
   bash parallel.sh
   ```

   This will:
   - Start a `scripts/aggregator.py` Flask server in a tmux session.
   - Launch `INSTANCES` tmux sessions running `python scripts/main.py ... --parallel` on different GPUs.
   - Periodically log aggregate metrics to Weights & Biases via the aggregator.

   Requirements: `tmux` installed and `wandb` configured (`wandb login`).

2. **Manual parallel runs + aggregator**

   Start the aggregator server directly (e.g. for ObjectNav):

   ```bash
   python scripts/aggregator.py --name ObjectNav_ours --sleep 10 --port 2000
   ```

   Then launch multiple workers, each with a distinct `--instance` id, sharing the same `--instances` and `--port`:

   ```bash
   # Example for a single worker (instance 0 of 4)
   python scripts/main.py \
     --config ObjectNav \
     --parallel \
     --instances 4 \
     --instance 0 \
     --port 2000 \
     --name ours_parallel
   ```

   Each worker will send metrics to the aggregator via `POST /log`, and the aggregator will log aggregate stats and GOAT-specific goal metrics to Weights & Biases.

### Tests

There is no dedicated automated test suite or standard test runner defined in this repo as of this writing.

## High-level architecture

### Overview

This project turns a vision-language model (VLM) into an end-to-end navigation policy in Habitat-Sim environments. The main flow is:

1. A CLI entrypoint (`scripts/main.py`) loads a YAML config from `config/` and resolves the requested `Env` and `Agent` classes.
2. The `Env` subclass (`GOATEnv` or `ObjectNavEnv` in `src/env.py`) manages dataset episodes, instantiates a `SimWrapper` (Habitat-Sim wrapper) and an `Agent`, and runs the episode loop.
3. The `Agent` subclass (`GOATAgent` or `ObjectNavAgent` in `src/agent.py`) implements the four-part VLMnav pipeline: **navigability → action proposal → projection → prompting**, calling into a `VLM` backend (`GeminiVLM` or `OpenAIVLM`) to choose actions.
4. `SimWrapper` (in `src/simWrapper.py`) converts high-level polar actions into Habitat-Sim agent motion and returns sensor observations.
5. `Env` logs step metadata, images, and metrics to disk; in parallel mode it also reports metrics to a Flask-based aggregator (`scripts/aggregator.py`), which in turn logs to Weights & Biases.

### Entry points & configuration

- **scripts/main.py**
  - Parses CLI args and loads `config/{name}.yaml`.
  - Merges selected CLI overrides into `env_cfg` (e.g. `num_episodes`, `max_steps`, `log_freq`, `instances`, `instance`, `parallel`, `port`, `name`).
  - Looks up `env_cls` from the global namespace (`GOATEnv` or `ObjectNavEnv`) and instantiates it with the merged config.
  - Calls `env.run_experiment()` to run all configured episodes.

- **Config files (`config/ObjectNav.yaml`, `config/GOAT.yaml`)** share a common structure:
  - `task`: label used in logging and aggregation (`"ObjectNav"` or `"GOAT"`).
  - `agent_cls`: name of the `Agent` subclass (`ObjectNavAgent` / `GOATAgent`).
  - `env_cls`: name of the `Env` subclass (`ObjectNavEnv` / `GOATEnv`).
  - `agent_cfg`: VLMnav agent hyperparameters (navigability mode, exploration bias, action spacing, VLM backend config, etc.).
  - `sim_cfg`: Habitat-Sim settings (scene config path, FOV, sensor height, agent radius/height, whether to use a goal-image agent, etc.).
  - `env_cfg`: experiment-level settings (episode counts, max steps, split, success thresholds, parallelization parameters, run name, port for the aggregator).

Changing the behavior of a run is usually done by editing the relevant YAML and/or overriding keys via CLI flags.

### Environment layer (`src/env.py`)

- **Env (base class)** encapsulates the common episode loop:
  - Initializes logging (to console or file depending on `env_cfg.parallel`).
  - Builds the `Agent` from `agent_cls` and `agent_cfg` and sets the default `PolarAction` for invalid VLM responses.
  - Provides `run_experiment()` which partitions the dataset among `instances` and iterates over episodes.
  - Manages per-episode state (`pandas` DataFrame for step metadata, accumulated agent distance, etc.).
  - Calls abstract hooks implemented by subclasses:
    - `_initialize_experiment()` to load datasets and build the list of episodes.
    - `_initialize_episode()` to construct the `SimWrapper`, set the initial agent state, and compute the initial shortest path.
    - `_step_env()` to integrate task-specific logic (goal definition, VLM agent stepping, metrics, logging, and termination).

- **GOATEnv**
  - Loads HM3D scenes and GOAT-Bench episodes from `data/datasets/goat_bench/hm3d/v1/{split}/content/*.gz`.
  - For each high-level GOAT episode, it builds a sequence of subgoals (object/description/image goals) and iterates through them.
  - For image goals, it uses `SimWrapper.get_goal_image` to render a goal image and attaches it to the observation for the agent.
  - After each subgoal, it uses SPL and success metrics to populate `wandb_log_data['task_data']['goal_data']` for the aggregator.

- **ObjectNavEnv**
  - Loads ObjectNav episodes from `data/datasets/objectnav_hm3d_v2/{split}/content/*.gz`.
  - For each episode, constructs a view-position set around the target object category and computes a shortest path using Habitat’s `MultiGoalShortestPath`.
  - On each step, attaches the object category string as `obs['goal']` for the agent.

Common functionality shared by both environments:
- `_calculate_metrics` uses Habitat pathfinding to compute distance-to-goal and SPL, and determines success vs. failure modes (`fp`, `max_steps`).
- `_log` writes per-step images to disk and a human-readable `details.txt`, and appends a row to the per-episode DataFrame.
- `_post_episode` saves `df_results.pkl` under `logs/{task}_{run_name}/{instance}/{episode_run}/`, resets the simulator and agent, optionally sends aggregated metrics to the Flask aggregator, and may trigger GIF creation via `create_gif`.

### Agent & navigation pipeline (`src/agent.py`)

- **Agent (base class)** defines the standard interface:
  - `step(obs) -> (agent_action, metadata)`
  - `get_spend()` for API-cost accounting.
  - `reset()` to clear per-episode state.

- **RandomAgent** is a simple baseline that samples random `PolarAction`s without any VLM.

- **VLMNavAgent** implements the core navigation algorithm common to GOAT and ObjectNav:
  - Holds the navigation state (voxel maps, exploration history, step index, initial position, last-turn-around step).
  - Initializes two VLM instances from `vlm_cfg`:
    - `actionVLM` for selecting the next action.
    - `stoppingVLM` for deciding when to stop.
  - Supports multiple navigability modes (`none`, `depth_estimate`, `segmentation`, `depth_sensor`) via `DepthEstimator` or `Segmentor` from `vlm.py`.
  - Optional **PIVOT** module (`pivot.py`) can replace the arrow-based projection step with sampled candidate points directly on the image when `agent_cfg.pivot` is enabled.
  - Key components:
    - **Navigability (`_navigability`)**: uses depth (either sensor or estimated) or segmentation-derived masks to determine navigable rays, updating a voxel occupancy map.
    - **Action proposer (`_action_proposer`)**: prunes and spaces navigable actions, introduces exploration bias based on previously explored areas in the voxel map.
    - **Projection (`_projection` / `_project_onto_image`)**: projects candidate actions into image space, draws red arrows, and annotates them with action numbers; adds a special "turn around" action 0 based on cooldown logic.
    - **Prompting (`_prompting`)**: builds text prompts conditioned on goal and projection mode, calls the VLM, parses JSON-like responses via `_eval_response`, and maps selected action numbers back to `PolarAction`s with `_action_number_to_polar`.
    - **Stopping (`_stopping_module`)**: calls the stopping VLM with one or more images (including the GOAT goal image when applicable) and decides whether to terminate based on the parsed `{'done': 0/1}` response.

- **GOATAgent** and **ObjectNavAgent** specialize `VLMNavAgent` for their tasks:
  - Implement `_choose_action` to build task-specific `goal` objects and wire in PIVOT vs. arrow-based prompting.
  - Implement `_construct_prompt` with different templates for `stopping`, `action`, `no_project`, and `pivot` modes, tailored to GOAT vs. ObjectNav.
  - GOATAgent includes `reset_goal()` to move between subgoals within a single high-level GOAT episode without resetting the entire voxel map.

### Simulation wrapper & action representation (`src/simWrapper.py`)

- **PolarAction** is the main low-level action primitive: `(r, theta, type)` where `r` is forward distance and `theta` is yaw rotation in radians.
  - Class attributes `default`, `stop`, and `null` are used to represent invalid-VLM fallbacks, termination, and "no-op / only observe" steps respectively.

- **SimWrapper** wraps Habitat-Sim to expose a simplified interface:
  - Initializes a simulator with RGB + depth sensors (and an optional second agent for goal-image rendering).
  - `step(PolarAction)` moves the Habitat agent by integrating rotation (`_rotate_yaw`) and forward motion (`_move_forward`) with collision-aware stepping via `pathfinder.try_step` or `try_step_no_sliding`.
  - Returns a dict of observations including `color_sensor`, `depth_sensor`, and `agent_state` (used extensively throughout the agent pipeline).
  - Provides utilities:
    - `get_goal_image` for GOAT image goals.
    - `set_state` to set the agent pose.
    - `get_path` as a thin wrapper over Habitat’s pathfinder to compute geodesic distances.

### VLM backends & perception modules (`src/vlm.py`)

- **VLM (base)** defines the interface used by navigation agents:
  - `call(images, prompt)` for stateless inference.
  - `call_chat(history, images, prompt)` for context-aware inference across steps.
  - `reset`, `rewind`, and `get_spend` hooks.

- **GeminiVLM**
  - Uses `google-generativeai` and the `GEMINI_API_KEY` env var.
  - Maintains a `GenerativeModel` chat session, tracks approximate spend using per-token pricing (flash vs. non-flash models), and resets/rewinds history based on the `history` parameter passed by the agent.

- **OpenAIVLM**
  - Uses `openai`-compatible `chat.completions` with optional custom `OPENAI_BASE_URL`.
  - Stores a simple message history list, truncated to `2 * history` turns.
  - Uses base64-encoded image URLs with MIME tags for vision input (see `utils.append_mime_tag`, `encode_image_b64`, `resize_image_if_needed`).

- **DepthEstimator** and **Segmentor** implement the optional navigability backends:
  - DepthEstimator wraps a HuggingFace `pipeline("depth-estimation")` model (e.g. ZoeDepth) and resizes predictions back to the RGB image size.
  - Segmentor wraps Mask2Former (`facebook/mask2former-swin-small-ade-semantic`) to compute semantic maps and extracts a floor/rug-based navigability mask.

### Utilities, logging, and visualization (`src/utils.py`, `scripts/aggregator.py`)

- `src/utils.py` provides:
  - Geometry transforms (`local_to_global`, `global_to_local`, `agent_frame_to_image_coords`, `depth_to_height`, etc.).
  - Image annotation helpers (`put_text_on_image`, drawing arrows and labels for actions and PIVOT samples).
  - Exception and traceback logging (`log_exception`).
  - GIF generation (`create_gif`) from per-step images saved under `logs/...`.

- `scripts/aggregator.py`:
  - Runs a Flask server exposing `/log` and `/terminate`.
  - Collects per-episode metrics from parallel workers, aggregates them across instances, and periodically logs summary statistics and task-specific metrics (e.g. GOAT goal success and SPL) to Weights & Biases.
  - A background thread (`wandb_logging`) handles periodic logging until a termination signal is received.

## Practical tips for new changes

- To add a new navigation **task**:
  - Create a new `Env` subclass in `src/env.py` (or a new module) following the `GOATEnv`/`ObjectNavEnv` pattern.
  - Add a matching `Agent` subclass in `src/agent.py` or elsewhere, and reference it via `agent_cls` in a new YAML under `config/`.
  - Point `env_cls` in the new YAML to the new environment class, then run via `python scripts/main.py --config <NewTaskName>`.

- To plug in a different **VLM backend**:
  - Implement a new `VLM` subclass in `src/vlm.py` (or another module), matching the `VLM` interface.
  - Update `agent_cfg.vlm_cfg.model_cls` and `model_kwargs` in the appropriate YAML config.
