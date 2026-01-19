# Integrating Answer Set Programming and Large Language Models for Reliable Structured Knowledge Extraction

Answer Set Programming (ASP) and Large Language Models (LLMs) offer complementary strengths in symbolic reasoning and natural language understanding. In this work, we present an integrated framework for reliable structured knowledge extraction from natural language that combines the syntactic capabilities of LLMs with the semantic constraints provided by ASP. Structured representations are obtained by guiding LLM generation through domain specific prompt templates and background knowledge specified in a declarative YAML configuration, and by validating and completing the extracted information using an ASP knowledge base. Building on this framework, we introduce grammar constrained decoding as a principled mechanism to enforce structural and semantic alignment between LLM outputs and target symbolic representations. By leveraging formal grammars, we gain fine grained control over the generation process, substantially reducing hallucinations and limiting nnecessary verbosity, while preserving extraction quality. We study three grammar formats commonly used to represent structured knowledge, namely CSV, Datalog, and JSON, and analyze their impact on correctness, efficiency, and computational cost.We evaluate the proposed approach on benchmarks derived from ASP Competitions. The results show that the integration of ASP consistently improves extraction accuracy over vanilla LLMs, particularly for smaller models. Moreover, grammar constrained decoding maintains these gains while significantly improving generation efficiency. Among the considered formats, CSV provides the best trade-off between accuracy and cost, JSON achieves the highest extraction quality at the expense of increased verbosity, and Datalog exhibits lower robustness in the extraction process. Overall, the results demonstrate that combining ASP with grammar constrained LLM output yields a reliable, scalable, and cost-effective approach to structured knowledge extraction.

## Run Experiments

Install [poetry](https://python-poetry.org/docs/#installation).

Inside the project folder run:

```bash
poetry shell
```

Then:

```bash
poetry install
```

There is CLI with for running the experiments for llmasp:

```bash
run-experiment --help
```

For further help with a single command use:

```bash
run-experiment <required_options> <command> --help
```

### Prompt Strategies (A1 vs A2)

The experiments rely on two main prompting strategies. You must configure the command flags accordingly:
    - A1:
        For LLMASP: Add the flag -a1 (or --a1_prompt).
        For Vanilla LLM: Use prompt level 1 (-pl 1).
    - A2:
        For LLMASP: This is the default (do not use -a1).
        For Vanilla LLM: Use higher prompt levels (e.g. -pl 10).

### LLMASP Experiments

1. Standard LLMASP (A2)

```bash
run-experiment -m <model_name> -s <server> -bf specifications/behaviors/behavior_second_report_v4.yml -af specifications/applications llmasp-full-test
```

2. Grammar Constrained LLMASP

To use the grammar constrained version, you need an Ollama instance running on a server that supports grammar constraints. Specify the output format (csv, json, datalog, etc.) using -otf.

```bash
run-experiment -m <model_name> -s <server> -bf specifications/behaviors/behavior_second_report_v6_{FORMAT}.yml -af specifications/applications llmasp-full-test -otf <FORMAT>
```

(Add -a1 to the command above to use the A1 prompt strategy).

Note on Knowledge Base: To run experiments involving the ASP Knowledge Base completion, append /asp to the application path (e.g., -af specifications/applications/asp).

### LLM Experiments

1. Standard LLM

```bash
run-experiment -m <model_name> -s <server> -bf specifications/behaviors/behavior_second_report_v4.yml -af specifications/applications  llama-full-test -pl <prompt_level>
```

2. Grammar Constrained LLM

```bash
run-experiment -m <model_name> -s <server> -af specifications/applications ollama-format-full-test -otf <FORMAT> -pl <prompt_level>
```

### Single Instance Execution

- For running a single experiment or a subset, replace *-full-test with *-run-selected and specify the problem name and quantity.

```bash
run-experiment -m llama3.1:70b -bf specifications/behaviors/behavior_second_report_v6.yml llmasp-run-selected -pn "Graph Coloring" -q 1
```

### Energy and Performance Tracking

The framework includes a built-in mechanism to monitor execution time and GPU energy consumption during experiments (If you have AMD GPUs). This utilizes rocm-smi to track energy usage (in Joules/uJ).

Options:
1. -elf / --energy-log-file: Path to the JSON file where energy stats will be saved.
2. -gpu / --gpu-index: Index of the GPU to monitor (default: 0). If None, all gpus monitored

To run an experiment and save the energy consumption data to energy_stats.json monitoring GPU 1:

```bash
run-experiment -m llama3.1:70b -bf specifications/behaviors/behavior_second_report_v6.yml llmasp-full-test -elf results/energy_stats.json -gpu 1
```


## Authors

* Mario Alviano
* Matteo Capalbo
* Georg Gottlob
* Lorenzo Grillo
* Irfan Kareem 
* Fabrizio Lo Scudo
* Sebastiano Piccolo
* Luis Angel Rodriguez Reiners
