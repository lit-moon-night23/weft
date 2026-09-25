# Weft

**An open-source simulator, energy model, and benchmarking harness for spatial compute-in-memory (CIM) accelerators, with a PyTorch frontend that lets algorithm researchers measure latency, energy, throughput, and accuracy without physical hardware.**

> Status: **scaffold (v0.0.1)**. The package skeleton exists, but no simulator logic is implemented yet. This README is the project's founding document. It sets out the problem, the scope, the architecture, the modeling methodology, and the roadmap. Sections marked *planned* describe interfaces that do not exist yet.

---

## Table of contents

1. [Problem statement](#1-problem-statement)
2. [Why existing tools are not enough](#2-why-existing-tools-are-not-enough)
3. [Goals and non-goals](#3-goals-and-non-goals)
4. [Target architecture class](#4-target-architecture-class)
5. [System overview](#5-system-overview)
6. [Modeling methodology](#6-modeling-methodology)
7. [Frontend: what a researcher writes](#7-frontend-what-a-researcher-writes)
8. [Hardware description format](#8-hardware-description-format)
9. [Outputs and reports](#9-outputs-and-reports)
10. [Benchmark suite: WeftBench](#10-benchmark-suite-weftbench)
11. [Validation and trust](#11-validation-and-trust)
12. [Repository layout](#12-repository-layout)
13. [Roadmap](#13-roadmap)
14. [How to cite](#14-how-to-cite)
15. [Contributing](#15-contributing)
16. [License](#16-license)
17. [Glossary](#17-glossary)

---

## 1. Problem statement

### 1.1 The situation

Deep learning workloads are now limited more by **data movement** than by arithmetic. Moving an operand from DRAM costs roughly two to three orders of magnitude more energy than the multiply-accumulate (MAC) it feeds. Two families of architectures attack this directly:

- **Compute-in-memory (CIM)** performs MACs inside or beside the memory array, for example analog current summation on SRAM or ReRAM crossbars, or bit-serial digital logic next to SRAM bitcells. Weights stay put, so weight movement nearly disappears.
- **Spatial dataflow architectures** tile many processing elements across a chip, connect them with an on-chip network, and stream activations between them so intermediate data rarely leaves the chip.

The most promising designs combine both: a **2-D array of CIM tiles** connected by a network-on-chip (NoC), with a whole model (or a large slice of it) pinned in on-chip weight memory and executed as a spatial pipeline.

### 1.2 The problem

Algorithm researchers increasingly want to co-design for this hardware. They propose quantization schemes, pruning patterns, noise-aware training, layer-fusion strategies, mixed-precision policies, and model architectures that "fit" CIM. To publish credible results they must answer:

> *"On a realistic CIM accelerator, what is the latency, energy, throughput, and accuracy of my model, and how does it compare to the baseline?"*

Today they cannot answer this well, because:

1. **No hardware access.** CIM chips are research prototypes. Almost no algorithm group can run a model on one.
2. **Fragmented tools.** Performance, energy, and accuracy are modeled by separate tools with incompatible inputs. A researcher must hand-translate a PyTorch model into a layer table for one tool, re-describe the hardware for another, and inject noise with a third. The three rarely agree on the same hardware.
3. **Layer-level, not model-level.** Most mappers evaluate one layer at a time. Spatial CIM performance is dominated by **inter-layer effects**: pipelining, weight-capacity limits that force reloads, NoC contention, and load imbalance across tiles. A per-layer sum misses these effects.
4. **Accuracy is decoupled from cost.** In analog CIM, the knobs that save energy (lower ADC precision, larger arrays, fewer bits per cell) directly add error. A tool that reports energy without the matching accuracy result invites misleading papers.
5. **Unreproducible numbers.** Papers report energy with private spreadsheets and unstated technology assumptions. Two papers claiming "5x better efficiency" usually cannot be compared.
6. **Poor support for modern models.** Transformers and LLMs add dynamic shapes, attention with activation-activation matmuls (poorly suited to weight-stationary CIM), KV caches, and softmax and normalization layers. Most CIM tools were built for CNNs.

### 1.3 The problem statement, in one paragraph

> Hardware-software co-design research for compute-in-memory and spatial dataflow accelerators lacks a **shared, open, validated substrate**: a single tool that takes an unmodified PyTorch model and a declarative hardware description, and reports **end-to-end latency, energy, throughput, area, and task accuracy** under a consistent and documented set of technology assumptions. Without it, algorithm researchers cannot evaluate hardware-aware ideas, hardware researchers cannot evaluate against real workloads, and results across papers are not comparable. **Weft aims to be that substrate.**

### 1.4 Success criteria

Weft succeeds when:

- An algorithm researcher with no hardware background goes from `pip install` to a latency/energy/accuracy report for their own PyTorch model in **under 30 minutes**.
- Its predictions are **validated against published silicon measurements**, with error bars published in the repository.
- Its benchmark results are **reproducible bit-for-bit** from a config file and a version tag.
- Co-design papers **cite Weft** as the platform on which their numbers were measured, and published configs let reviewers rerun them.

---

## 2. Why existing tools are not enough

Weft builds on a strong body of prior work and should interoperate with it where possible. The gap is in the combination, not in any single capability.

| Tool | Strength | What is missing for this use case |
|---|---|---|
| Timeloop + Accelergy | Rigorous per-layer mapping search and component energy estimation | Per-layer; no inter-layer pipelining on a tile array; no accuracy model; manual workload entry |
| CiMLoop | Extends Timeloop to CIM with detailed analog circuit energy models | Inherits per-layer scope; accuracy under noise is outside the tool |
| DNN+NeuroSim | Circuit-level CIM energy/area plus device non-ideality effects on accuracy | Fixed hierarchical architecture template; limited transformer support; not built around arbitrary PyTorch graphs |
| MAESTRO, ZigZag | Fast analytical dataflow cost models | Digital accelerators first; no analog CIM accuracy model |
| SCALE-Sim | Cycle-accurate systolic array simulation | Systolic arrays only; no CIM; no NoC-level multi-layer mapping |
| IBM AIHWKit | High-quality analog noise models for training and inference | Accuracy only; no latency, energy, or mapping |
| gem5-based simulators | Full-system detail | Too slow and too low-level for algorithm researchers |

**Weft's position:** one pipeline from `torch.nn.Module` to a joint **(latency, energy, throughput, area, accuracy)** report. Its focus is whole-model mapping onto a spatial array of CIM tiles. Where a mature component model exists (for example Accelergy energy tables or AIHWKit noise models), Weft should import it rather than reinvent it.

---

## 3. Goals and non-goals

### Goals

- **G1. No-hardware workflow.** Run any exportable PyTorch model with a single Python call or CLI command.
- **G2. Joint metrics.** Report latency, energy, throughput, area, and accuracy from one consistent hardware description.
- **G3. Whole-model spatial mapping.** Model layer placement, pipelining, weight reloads, and NoC traffic across a tile array.
- **G4. Multiple fidelity levels.** A fast analytical mode for design-space sweeps and a slower event-driven mode for detailed studies. Both use the same inputs.
- **G5. Validated.** Calibrate against published silicon and publish the error for every validation point.
- **G6. Reproducible.** Every result is keyed to a Weft version, a hardware config hash, a workload hash, and a random seed.
- **G7. Extensible.** New tile types, devices, ADCs, NoCs, and mappers are plug-ins with a documented interface.
- **G8. Citable benchmarks.** Ship a versioned benchmark suite and reference hardware configs that papers can name directly.

### Non-goals

- **Not RTL.** Weft does not generate Verilog or model gate-level timing.
- **Not a SPICE substitute.** Circuit-level parameters are inputs, taken from literature or from a user's own circuit simulations.
- **Not a training framework.** Weft evaluates models. Noise-aware training can call Weft's noise models, but the training loop belongs to the user.
- **Not a general-purpose CPU/GPU simulator.** GPU numbers appear only as reference baselines from external measurements.

---

## 4. Target architecture class

Weft models a parameterized template, the **Spatial CIM Array (SCA)**.

```
 ┌──────────────────────────── Chip ─────────────────────────────┐
 │  Global buffer (SRAM)  ◄──►  Off-chip interface (DRAM / HBM)   │
 │        ▲                                                       │
 │        │  NoC (mesh / torus / hierarchical)                    │
 │  ┌─────┴─────┬───────────┬───────────┬───────────┐             │
 │  │  Tile     │  Tile     │  Tile     │  Tile     │   ...       │
 │  ├───────────┼───────────┼───────────┼───────────┤             │
 │  │  Tile     │  Tile     │  SIMD/    │  Tile     │   ...       │
 │  │           │           │  vector   │           │             │
 │  └───────────┴───────────┴───────────┴───────────┘             │
 └────────────────────────────────────────────────────────────────┘

 Tile = router + local buffers + N CIM macros + digital post-processing
 CIM macro = array (rows x cols) + DACs/drivers + ADCs/sense amps + shift-add
```

**Configurable dimensions:**

| Level | Parameters |
|---|---|
| Device / cell | SRAM (6T/8T/10T), ReRAM, PCM, FeFET, MRAM; bits per cell; conductance range; on/off ratio; variation; drift; read noise |
| Macro | Array rows and columns; analog vs digital CIM; rows activated per cycle; input bit-serialization; ADC type, bits, and sharing ratio; weight bit-slicing |
| Tile | Number of macros; input/output buffer sizes; accumulator width; activation function unit; tile-local SIMD |
| Array | Tile count and grid shape; heterogeneous tiles (for example CIM tiles plus digital vector tiles for softmax and normalization) |
| Interconnect | Topology, link width, router latency, flit size, energy per bit-hop |
| Memory | Global buffer size and bandwidth; DRAM/HBM bandwidth and energy per bit |
| Technology | Node (for example 22 nm, 28 nm, 7 nm), supply voltage, clock frequency, with scaling rules documented per component |

**Supported execution styles:**

- **Weight-stationary, fully resident.** The whole model fits on chip and layers form a spatial pipeline.
- **Weight-stationary with reload.** Layers are grouped into segments and weights are re-programmed between segments. Write energy, write latency, and endurance are all modeled.
- **Hybrid.** Activation-activation operations (attention `QK^T`, `AV`) run on digital tiles or on CIM with dynamic weight writes.

---

## 5. System overview

```
  PyTorch model ─┐
                 ▼
        ┌─────────────────┐     ┌─────────────────────┐
        │ 1. Frontend     │     │  Hardware config     │
        │ torch.export →  │     │  (YAML / Python)     │
        │ Weft IR         │     └─────────┬───────────┘
        └────────┬────────┘               │
                 ▼                        ▼
        ┌──────────────────────────────────────────┐
        │ 2. Compiler / mapper                     │
        │  - operator lowering & quantization      │
        │  - weight tiling onto macros             │
        │  - layer → tile placement                │
        │  - pipeline scheduling, NoC routing      │
        └────────────────────┬─────────────────────┘
                             ▼  Mapped program
        ┌──────────────────────────────────────────┐
        │ 3. Simulation engines                    │
        │  a. Analytical (seconds)                 │
        │  b. Event-driven (minutes)               │
        │  c. Functional / accuracy (noise-aware)  │
        └────────────────────┬─────────────────────┘
                             ▼
        ┌──────────────────────────────────────────┐
        │ 4. Energy & area model (component library)│
        └────────────────────┬─────────────────────┘
                             ▼
        ┌──────────────────────────────────────────┐
        │ 5. Reports: JSON, CSV, HTML dashboard,   │
        │    per-layer / per-tile breakdowns,      │
        │    roofline and Pareto plots             │
        └──────────────────────────────────────────┘
```

### 5.1 Frontend

- Captures the graph with `torch.export`. The ONNX importer is a fallback.
- Lowers the graph to the **Weft IR**, a small, typed graph of tensor operators: `matmul`, `conv`, `elementwise`, `reduce`, `softmax`, `norm`, `reshape`, `kv_read`, `kv_write`.
- Records a quantization spec for every tensor: bits, scheme, and granularity. The spec is read from the model (for example PyTorch AO quantized modules) or supplied by the user.

### 5.2 Compiler and mapper

- **Lowering.** Maps each IR operator to a hardware operation class: CIM-MVM, digital vector op, or data movement.
- **Weight tiling.** Splits each weight matrix across macro rows and columns and across bit-slices, so that ADC resolution and array size constraints are satisfied.
- **Placement.** Assigns tiles to layers. Built-in strategies include greedy, balanced-throughput (minimizing the slowest pipeline stage), and ILP/simulated annealing for small arrays. The mapper interface is pluggable, so researchers can publish their own mappers.
- **Scheduling.** Produces a pipeline schedule with stage boundaries, buffer allocations, and NoC flows.

### 5.3 Simulation engines

| Engine | Speed target | Use |
|---|---|---|
| **Analytical** | < 1 s per model/config | Design-space exploration; closed-form throughput from stage bottlenecks; NoC traffic as volume divided by bisection bandwidth |
| **Event-driven** | Seconds to minutes | Contention, pipeline fill and drain, buffer stalls, and batch/sequence effects, simulated at the level of macro operations and NoC flits (not gates) |
| **Functional / accuracy** | Similar to a PyTorch inference run | Runs the actual model with CIM non-idealities injected at every mapped MVM and reports the task metric |

The analytical engine must stay within a documented error band of the event-driven engine on every benchmark. CI enforces this.

### 5.4 Energy and area model

- A **component library**: each component (ADC, DAC, array read, array write, SRAM buffer access, router hop, DRAM access, vector ALU op) has an energy-per-action and an area. Each entry is parameterized by technology node, voltage, and precision.
- Every entry carries **provenance**: the source paper, table or figure, original node, and scaling rule applied. Values without provenance are rejected by CI.
- Accelergy-compatible import and export, so energy tables can be shared with the Timeloop ecosystem.

---

## 6. Modeling methodology

### 6.1 Latency and throughput

For a macro performing a matrix-vector multiply (MVM) with `b_in`-bit inputs applied bit-serially, `R` rows activated per cycle, and ADCs shared across `s` columns:

```
t_MVM ≈ b_in × ceil(rows / R) × s × t_ADC  +  t_shift_add  +  t_buffer
```

At the tile level, time is the maximum over macros plus local reduction. At the array level, the steady-state pipeline throughput is set by the slowest stage:

```
throughput ≈ 1 / max_stage( t_compute(stage) , t_noc_in(stage) , t_noc_out(stage) )
latency    ≈ Σ_stages t_stage  +  pipeline fill  +  off-chip transfer
```

The event-driven engine replaces these closed forms with explicit simulation wherever contention or buffering matters.

### 6.2 Energy

```
E_total = Σ_components  N_actions(component) × E_per_action(component, node, V, precision)
        + P_leakage × latency
```

Action counts come from the mapped program and do not depend on which engine produced the timing. As a result, the analytical and event-driven engines report identical dynamic energy, and only leakage differs.

### 6.3 Accuracy under non-idealities

For analog CIM, each mapped MVM is evaluated as:

```
y = ADC_q( Σ_i  x_i · (G_ij + δ_prog + δ_read(t) + δ_drift(t)) + IR_drop(x, G) ) → digital shift-add
```

- **Programming variation** is sampled once per weight at "deployment" and fixed across inferences.
- **Read noise** is resampled on every read.
- **Drift** is a function of time since programming, which matters for PCM and ReRAM.
- **ADC quantization and clipping** use the configured bits and range.
- **IR drop** has a first-order analytical model, with an option for a more detailed solver.
- Models are pluggable. An AIHWKit adapter is planned so that users can reuse its calibrated device models.

Accuracy results are reported as **mean ± std across N device-sample seeds**, never as a single run.

### 6.4 Fidelity contract

Every reported number carries a **fidelity tag** (`analytical`, `event`, `functional`) and a **calibration status** (`validated`, `extrapolated`, `uncalibrated`). A result is `extrapolated` when the config lies outside the envelope of validated points. Reports make these tags visible, so readers of a paper can see how much to trust each number.

---

## 7. Frontend: what a researcher writes

*Planned API.*

### 7.1 Python

```python
import torch, torchvision
import weft

model = torchvision.models.resnet18(weights="DEFAULT").eval()
example = torch.randn(1, 3, 224, 224)

hw = weft.hardware.load("presets/sram-dcim-22nm-64tile.yaml")   # or build in Python

report = weft.evaluate(
    model,
    example_inputs=(example,),
    hardware=hw,
    quant=weft.quant.uniform(weights=4, activations=8),
    engine="analytical",            # "analytical" | "event"
    accuracy=weft.accuracy.ImageNetVal(n_samples=5000, seeds=5),  # optional
)

print(report.summary())
# latency, energy/inference, throughput, TOPS/W, area, top-1 (mean ± std),
# each tagged with fidelity and calibration status

report.to_html("resnet18_report.html")
report.to_json("resnet18_report.json")
```

### 7.2 Design-space sweep

```python
sweep = weft.Sweep(
    model, example_inputs=(example,),
    base_hardware=hw,
    axes={
        "macro.adc.bits":   [4, 5, 6, 8],
        "macro.rows":       [128, 256, 512],
        "array.tiles":      [16, 64, 256],
    },
    engine="analytical",
)
df = sweep.run(workers=8)      # pandas DataFrame
weft.plot.pareto(df, x="energy_per_inference", y="top1")
```

### 7.3 CLI

```bash
weft run   --model hf:bert-base-uncased --seq-len 128 \
           --hw presets/reram-acim-28nm.yaml --engine event --out results/
weft bench --suite weftbench-v1 --hw presets/ --out results/   # full benchmark suite
weft validate                                                  # re-run silicon validation points
```

### 7.4 Custom hooks for co-design research

- **Custom quantizer.** Supply a per-tensor bit-width or format policy.
- **Custom mapper.** Subclass `weft.mapping.Mapper` to publish a new placement algorithm.
- **Custom noise model.** Subclass `weft.noise.DeviceModel`.
- **Differentiable proxy (stretch goal).** The analytical engine exposes smooth cost estimates so hardware cost can be added to a training loss.

---

## 8. Hardware description format

*Planned schema.* Hardware is described declaratively in YAML and validated against a JSON Schema. The same object can be built in Python.

```yaml
weft_version: "0.1"
name: sram-dcim-22nm-64tile
technology: { node_nm: 22, vdd: 0.8, freq_mhz: 500 }

macro:
  kind: digital_cim            # analog_cim | digital_cim
  cell: sram_8t
  rows: 256
  cols: 256
  weight_bits: 4
  input_bits: 8
  input_mode: bit_serial
  rows_active: 256
  adc: null                    # digital CIM uses adder trees
  adder_tree: { bits: 12 }

tile:
  macros: 8
  input_buffer_kib: 32
  output_buffer_kib: 32
  vector_unit: { lanes: 32, ops: [relu, gelu, add, mul] }

array:
  grid: [8, 8]
  special_tiles:
    - { kind: digital_vector, count: 4, ops: [softmax, layernorm] }

noc:  { topology: mesh, link_bits: 256, router_cycles: 2 }
memory:
  global_buffer_mib: 4
  offchip: { kind: lpddr5, bandwidth_gbps: 51.2 }

energy_library: default-22nm   # every entry carries provenance
```

Reference presets ship in `presets/`, and each preset states which published design it approximates. **Presets are named approximations of published designs, not claims of exact reproduction.**

---

## 9. Outputs and reports

Each run produces:

- **`summary.json`**: headline metrics, fidelity tags, calibration status, and all input hashes.
- **`layers.csv`**: per-layer latency, energy by component, macro utilization, and assigned tiles.
- **`tiles.csv`**: per-tile utilization, idle time, and NoC ingress and egress.
- **`energy_breakdown.csv`**: energy by component class (ADC, array, buffers, NoC, DRAM, leakage).
- **`report.html`**: an interactive dashboard with a tile heatmap, pipeline Gantt chart, energy breakdown, and roofline position.
- **`manifest.json`**: Weft version, git SHA, config hash, workload hash, seeds, and host info. This file is what makes a result reproducible.

---

## 10. Benchmark suite: WeftBench

A versioned, frozen set of workloads and configs for papers to report against. Proposed contents for `weftbench-v1`:

| Category | Workloads |
|---|---|
| CNN | ResNet-18/50, MobileNetV2, EfficientNet-B0 |
| Transformer encoder | BERT-base (seq 128/512), ViT-B/16 |
| LLM decode | Llama-class 1B and 7B at batch 1, prefill and decode reported separately |
| Recommendation / other | DLRM (MLP portion), a small speech model |
| Micro-benchmarks | Single MVM sweeps, GEMM shape sweeps, NoC stress patterns |

Each benchmark reports a fixed metric set: latency, energy per inference, throughput, TOPS/W at the model level (not peak), area, and task accuracy. A public leaderboard of hardware configs and mapping or quantization techniques is a stretch goal.

---

## 11. Validation and trust

A simulator becomes a citable substrate only if people trust it. The plan:

1. **Component-level validation.** Compare macro-level energy and latency against published CIM macro measurements from ISSCC, VLSI, and JSSC papers. Each comparison records the paper, the figure or table, and the percent error.
2. **System-level validation.** Compare full-model results against published chip-level measurements wherever authors report end-to-end numbers.
3. **Cross-tool validation.** Compare per-layer results against Timeloop/CiMLoop and NeuroSim on shared configurations, and explain any divergence.
4. **Accuracy validation.** Reproduce published accuracy-under-noise results, for example from AIHWKit studies, within their reported variance.
5. **Internal consistency.** CI checks analytical against event-driven engines, energy conservation (component sum equals total), and determinism under fixed seeds.
6. **Public error table.** `docs/validation.md` lists every validation point with its error. Weft makes no accuracy claims beyond what that table shows.

No validation numbers exist yet. They will be added as each point is completed.

---

## 12. Repository layout

The skeleton below exists. Every module is currently an empty placeholder with a docstring stating its responsibility.

```bash
pip install -e ".[dev]"   # add ,torch for the PyTorch frontend once it lands
pytest
weft --version
```

```
weft/
├── README.md                ← this document
├── LICENSE                  # Apache-2.0
├── CITATION.cff
├── CONTRIBUTING.md
├── pyproject.toml
├── weft/
│   ├── frontend/            # torch.export / ONNX → Weft IR
│   ├── ir/                  # IR definitions, passes, quantization specs
│   ├── hardware/            # config schema, loaders, presets API
│   ├── mapping/             # tiling, placement, scheduling, mapper plug-ins
│   ├── engines/
│   │   ├── analytical/
│   │   ├── event/           # event-driven core (Python first, C++/Rust core later)
│   │   └── functional/      # noise-injected PyTorch execution
│   ├── energy/              # component library + provenance + Accelergy adapter
│   ├── noise/               # device, ADC, and IR-drop models; AIHWKit adapter
│   ├── report/              # JSON/CSV/HTML writers, plots
│   └── cli.py
├── presets/                 # reference hardware configs (YAML)
├── benchmarks/weftbench/    # frozen workloads + runner
├── validation/              # silicon-comparison scripts and data
├── examples/                # notebooks: quickstart, sweep, custom mapper, noise-aware eval
├── tests/
└── docs/                    # user guide, modeling reference, validation table, API docs
```

---

## 13. Roadmap

| Milestone | Scope | Exit criterion |
|---|---|---|
| **M0: Foundations** | IR, hardware schema, `torch.export` frontend, component energy library with provenance | ResNet-18 lowers to IR and a preset config validates |
| **M1: Analytical MVP** | Weight tiling, greedy placement, analytical engine, JSON/CSV reports | `weft.evaluate` works end to end on CNNs and BERT-base |
| **M2: Accuracy** | Functional engine with analog noise and ADC models, multi-seed accuracy | Joint energy–accuracy Pareto sweep in an example notebook |
| **M3: Event-driven engine** | NoC and buffer contention, pipeline fill and drain, LLM prefill and decode | Analytical and event engines agree within a documented band |
| **M4: Validation v1** | At least 5 macro-level and 2 system-level silicon comparisons | `docs/validation.md` published |
| **M5: WeftBench v1 + paper** | Frozen suite, presets, HTML dashboard, tool paper and artifact | Tagged v1.0 release, DOI via Zenodo, artifact-evaluation badges targeted |
| **Later** | Differentiable cost proxy, performance core in Rust/C++, multi-chip scaling, sparsity support, training-time CIM | Driven by community demand |

---

## 14. How to cite

Once v1.0 is released, a `CITATION.cff` and a Zenodo DOI will be provided. Until then, placeholder:

```bibtex
@software{weft2026,
  title   = {Weft: An Open Simulator and Benchmark Suite for Spatial Compute-in-Memory Accelerators},
  author  = {Shivam, Pradipto and contributors},
  year    = {2026},
  url     = {TBD},
  version = {0.0}
}
```

Papers that use Weft should report the **Weft version**, the **hardware config** (or preset name and hash), the **engine** used, and the **calibration status** of the reported numbers. `manifest.json` contains all four.

---

## 15. Contributing

Contributions are welcome in these areas:

- **Energy and area data.** New component entries must include a citation, the source node, and the scaling rule used.
- **Presets.** Configs that approximate published chips, with a note on what was approximated.
- **Mappers and noise models.** Implemented as plug-ins with tests.
- **Validation points.** Comparisons against silicon measurements.
- **Workloads.** New models for WeftBench, with a frozen export and a fixed input set.

Every PR must pass the unit tests, the engine-consistency checks, and the provenance lint. See `CONTRIBUTING.md`.

---

## 16. License

Apache License 2.0 (proposed). It is permissive for academic and industrial use and includes an explicit patent grant.

---

## 17. Glossary

| Term | Meaning |
|---|---|
| **CIM** | Compute-in-memory: performing MAC operations inside or beside memory arrays |
| **ACIM / DCIM** | Analog / digital compute-in-memory |
| **MVM** | Matrix-vector multiplication, the core CIM operation |
| **ADC / DAC** | Analog-to-digital / digital-to-analog converter |
| **NoC** | Network-on-chip connecting tiles |
| **Spatial dataflow** | Executing a model by placing operators on physically distinct compute units and streaming data between them |
| **Weight-stationary** | Weights stay in place while activations move |
| **IR drop** | Voltage loss along crossbar wires that distorts analog MVM results |
| **Drift** | Time-dependent change in device conductance after programming |
| **Fidelity tag** | Label that states which engine produced a number |
| **Calibration status** | Label that states whether a number lies within Weft's validated envelope |
