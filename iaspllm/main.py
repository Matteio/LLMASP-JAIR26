import typer
import dataclasses
from dataclasses import asdict
import random
import requests
import os
from pathlib import Path
from itertools import groupby
from typing import Optional, Dict, Any, List
from dumbo_utils.console import console
import json
from rich.progress import Progress
from rich.table import Table
from rich.live import Live
from dumbo_asp.primitives.atoms import GroundAtom
from dumbo_asp.primitives.models import Model
from enum import Enum
import subprocess
import re
import time
import datetime
import csv

from .utils import load_data, save_results, evaluate_model, load_partial_results, parse_fact_line
from .llmasp import llm
from .llmasp.llm import UsageMetadata

ROOT_PATH = Path(__file__).parent.parent.resolve()

def get_gpu_index(server_url:str) -> Optional[int]:
    if server_url.endswith("11436"):
        return 0
    elif server_url.endswith("11437"):
        return 1
    else:
        return 2
    
def get_all_gpu_energy_uJ(gpu_index: int | None = None) -> dict[int, float]:
    try:
        result = subprocess.run(["rocm-smi", "--showenergy"], capture_output=True, text=True, check=True)
        output = result.stdout

        pattern = r"GPU\[(\d+)\].*?Accumulated Energy \(uJ\): ([\d\.]+)"
        matches = re.findall(pattern, output)

        energy_data = {int(gpu): float(energy) for gpu, energy in matches}

        if gpu_index is not None:
            if gpu_index in energy_data:
                return {gpu_index: energy_data[gpu_index]}
            else:
                print(f"Warning: GPU index {gpu_index} non trovato nei risultati.")
                return {}
            
        return energy_data
    
    except FileNotFoundError:
        console.log("Error: `rocm-smi` is not installed.")
        return {}
    
    except Exception as e:
        console.log(f"Error: {e}")
        return {}

def run_with_energy_tracking(command_name: str, gpu_index:int, log_file: Optional[Path], target_function, *args, **kwargs):
    console.log(f"[bold green] Run experiment '{command_name}' measuring energy and time [/bold green]")

    try:
        energy_before = get_all_gpu_energy_uJ(gpu_index)
        start_time = time.time()

        target_function(*args, **kwargs)

    except Exception as e:
        console.log(f"[red]Error: {e}[/red]")

    finally:       

        end_time = time.time()
        energy_after = get_all_gpu_energy_uJ(gpu_index)

        elapsed_time = end_time - start_time
        timestamp = datetime.datetime.now().isoformat()

        results = {
            "timestamp": timestamp,
            "command": command_name,
            "elapsed_time_s": round(elapsed_time, 2),
            "gpus": {}
        }

        for gpu_id in energy_before:
            if gpu_id in energy_after:
                energy_diff_uJ = energy_after[gpu_id] - energy_before[gpu_id]
                results["gpus"][gpu_id] = {
                    "energy_consumed_uJ": round(energy_diff_uJ, 0),
                    "energy_consumed_J": round(energy_diff_uJ / 1_000_000, 3)
                }
                
        for gpu_id, data in results["gpus"].items():
            console.log(f"[bold]GPU[{gpu_id}]:[/bold] {data['energy_consumed_J']} Joule ({int(data['energy_consumed_uJ'])} uJ)")

        if log_file:
            try:
                log_file.parent.mkdir(parents=True, exist_ok=True)

                if log_file.exists():
                    with open(log_file, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                else:
                    existing_data = []

                existing_data.append(results)

                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, indent=2)

                console.log(f"Results saved in {log_file}")
            except Exception as e:
                console.log(f"[red]Error during the save: {e}[/red]")

@dataclasses.dataclass(frozen=True)
class AppOptions:
    model: str
    server: str
    ollama_key: str
    behavior_file: Path
    application_file: Path
    dataset: Path
    single_pass: bool
    output_file: Path
    stats_file: Path
    log_prompts: bool
    debug: bool
    pred: bool
    partial_file: Path
    gpu_index: Optional[int]
    energy_log_file: Optional[Path]

class Level(str, Enum):
    one = 1
    two = 2
    three = 3
    four = 4
    five = 5

class OllamaRequests:
    def __init__(self, model: str, server_url: str = "http://127.0.0.1:11436", api_key: str = "ollama"):
        self.model = model
        self.server_url = server_url
        self.api_key = api_key
        if self.server_url.endswith("/v1"):
            self.server_url = self.server_url[:-3]
        
    def call(self, prompt: str, format: Optional[str] = None, 
            temperature: float = 0, options: Optional[Dict[str, Any]] = None) -> tuple:

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options or {}
        }
        
        
        if format:
            payload["format"] = format
            
        if "temperature" not in payload["options"]:
            payload["options"]["temperature"] = temperature
            
        console.log(f"[yellow]Sending request to: {self.server_url}/api/generate")
        console.log(f"[yellow]Payload model: {payload['model']}")
        console.log(f"[yellow]Format: {format}")

        try:
            full_url = f"{self.server_url}/api/generate"
            console.log(f"[yellow]Full URL: {full_url}")
            
            headers = {}
            if self.api_key != "ollama":
                headers["Authorization"] = f"Bearer {self.api_key}"
                
            console.log(f"[yellow]Headers: {headers}")
            
            response = requests.post(
                full_url,
                headers=headers,
                json=payload,
                timeout=3600  #60 minutes timeout
            )
            
            console.log(f"[yellow]Response status code: {response.status_code}")
            
            if response.status_code != 200:
                console.log(f"[red]Error response: {response.text}")
                return f"Error: HTTP {response.status_code}", {}
                
            try:
                data = response.json()
                console.log("[green]Successfully parsed JSON response")
            except Exception as e:
                console.log(f"[red]Error parsing JSON: {str(e)}")
                console.log(f"[red]Response text: {response.text[:500]}...")
                return f"Error parsing response: {str(e)}", {}
            
            if format == "json":
                try:
                    completion = json.dumps(data.get("response", data), ensure_ascii=False, indent=2)
                except Exception as e:
                    console.log(f"[red]Errore durante il parsing JSON della risposta: {str(e)}")
                    completion = str(data)
            else:
                completion = data.get("response", "")
            
            console.log(f"[green]Got completion of length: {len(completion)}")

            metadata = UsageMetadata.from_ollama_dict(data)
            metadata = json.dumps(asdict(metadata))
            console.log(f"[green]Metadata: {metadata}")
            
            return completion, metadata
            
        except Exception as e:
            console.log(f"[red]Error calling Ollama API: {str(e)}")
            return "", {}

app_options: Optional[AppOptions] = None
app = typer.Typer()

def make_table(title, table_data):
    table = Table(title=title)
    for idx, col in enumerate(table_data[0]):
        table.add_column(col, justify="left" if idx == 0 else "right")
    for row in table_data[1:-1]:
        table.add_row(*row)
    table.add_row()
    table.add_row(*table_data[-1])
    return table

@app.callback()
def main(
        model: str = typer.Option(..., "--model", "-m", help="Ollama model ID"),
        server: str = typer.Option("http://localhost:11434/v1", "--server", "-s",
                                   envvar="OLLAMA_SERVER", help="Ollama server URL"),
        ollama_key: str = typer.Option("ollama", "--ollama-key", "-ok", envvar="OLLAMA_API_KEY",
                                       help="Ollama API key"),
        behavior_file: Path = typer.Option(..., "--behavior-file", "-bf",
                                           help="Path for behavior file"),
        application_file: Path = typer.Option(..., "--application-file", "-af",
                                              help="Path for application file (or directory)"),
        dataset: Path = typer.Option(ROOT_PATH / "data/dataset.json", "--dataset", "-d", help="Path for dataset file"),
        single_pass: bool = typer.Option(False, "--single-pass", "-sp",
                                         help="Make single queries to llm"),
        output_file: Path = typer.Option(ROOT_PATH / "iaspllm/results/llm_output.json",
                                         "--output-file", "-of", help="Output file path for llm response"),
        stats_file: Path = typer.Option(ROOT_PATH / "iaspllm/results/llm_output_stats.csv",
                                        "--stats-file", "-sf", help="Statistics output file path for llm response"),
        log_prompts: bool = typer.Option(False, "--log-prompts", help="Log prompts and responses"),
        debug: bool = typer.Option(False, "--debug", help="Print debug error messages"),
        pred: bool = typer.Option(True, "--pred", help="If present we don't put predicate in the response, so we use a different method to evaluate"),
        partial_file: Path = typer.Option(ROOT_PATH / "iaspllm/results/llm_output.jsonl", "--partial-file", "-pf", help="Path for partial file"),
        gpu_index: Optional[int] = typer.Option(0, "--gpu-index", "-gpu", help="Index of the GPU to monitor for energy consumption."), 
        energy_log_file: Optional[Path] = typer.Option(None, "--energy-log-file", "-elf", help="Path to save the energy and time consumption log."), 

):
    """
    CLI for LLMASP
    """
    global app_options

    app_options = AppOptions(
        model=model,
        server=server,
        ollama_key=ollama_key,
        behavior_file=behavior_file,
        application_file=application_file,
        dataset=dataset,
        single_pass=single_pass,
        output_file=output_file,
        stats_file=stats_file,
        log_prompts=log_prompts,
        debug=debug,
        pred=pred,
        partial_file=partial_file,
        gpu_index=gpu_index,
        energy_log_file=energy_log_file,
    )

@app.command(name="llmasp-run-selected")
def llmasp_command_run_selected(
        problem_name: str = typer.Option(..., "--problem-name", "-pn", help="Problem name"),
        quantity: int = typer.Option(..., "--quantity", "-q", help="Instances quantity"),
        random_instances: bool = typer.Option(False, "--random-instances", "-rn",
                                              help="Take random instances"),
        output_format: str = typer.Option("json", "--output-format", "-otf", 
                                      help="Output format (json, csv, etc.)")
) -> None:
    """
    Run selected testcases with LLMASP.
    """
    assert app_options is not None

    with console.status("Loading data..."):
        data = load_data(app_options.dataset)
        problem_name = problem_name.replace(" ", "")
        problem_instances = [instance for instance in data
                             if instance["problem_name"].replace(" ", "") == problem_name]
        if random_instances:
            problem_instances = random.sample(problem_instances, quantity)
        else:
            problem_instances = problem_instances[:quantity]
        console.log("Data loaded!")
    console.log(f"Problem: {problem_name}; Instances: {quantity}")
    console.log(f"Testing model: {app_options.model}")
    results = test_problem(
        app_options.model,
        app_options.server,
        app_options.ollama_key,
        app_options.behavior_file,
        app_options.application_file,
        problem_instances,
        single_pass=app_options.single_pass,
        output_format = output_format
    )
    console.log("Done with the model")
    save_results(results, file_path=app_options.output_file)
    console.print('---------------------------------Evaluate responses--------------------------------------')
    evaluate_model(app_options.model, results, app_options.stats_file)
    console.print('----------------------------------------Done---------------------------------------------')

def _llmasp_command_full_test_worker(quantity: Optional[int], output_format: str, a1_prompt: bool):
    """Worker function for llmasp-full-test logic."""
    assert app_options is not None
    with console.status("Loading data..."):
        data = load_data(app_options.dataset)
        console.log("Data loaded!")
    if quantity:
        data.sort(key=lambda i: i["problem_name"])
        grouped = groupby(data, key=lambda i: i["problem_name"])
        data = [inst for _, group in grouped for inst in list(group)[:quantity]]
    console.log(f"Testing model: {app_options.model}")

    results = test_llmasp_dataset(
        app_options.model, 
        app_options.server, 
        app_options.ollama_key,
        app_options.behavior_file, 
        app_options.application_file, 
        data,
        single_pass=app_options.single_pass, 
        output_format=output_format,
        a1_prompt = a1_prompt
    )

    save_results(results, file_path=app_options.output_file)
    console.print('---------------------------------Evaluate responses--------------------------------------')
    evaluate_model(app_options.model, results, app_options.stats_file)
    console.print('----------------------------------------Done---------------------------------------------')

@app.command(name="llmasp-full-test")
def llmasp_command_full_test(
    quantity: Optional[int] = typer.Option(None, "--quantity", "-q", help="max number of instance for problem"),
    output_format: str = typer.Option(None, "--output-format", "-otf", help="output format (json, csv, ecc.)"),
    a1_prompt: bool = typer.Option(False, "--a1_prompt", "-a1", help="use only format in the prompt")
):
    """Run for the entire dataset with LLMASP."""
    assert app_options is not None
    if app_options.gpu_index is not None:
        run_with_energy_tracking(
            command_name="llmasp-full-test",
            gpu_index=app_options.gpu_index,
            log_file=app_options.energy_log_file,
            target_function=_llmasp_command_full_test_worker,
            quantity=quantity,
            output_format=output_format,
            a1_prompt = a1_prompt
        )
    else:
        _llmasp_command_full_test_worker(quantity, output_format, a1_prompt)

@app.command(name="llama-run-selected")
def llama_command_run_selected(
        problem_name: str = typer.Option(..., "--problem-name", "-pn", help="Problem name"),
        quantity: int = typer.Option(..., "--quantity", "-q", help="Instances quantity"),
        prompt_level: Level = typer.Option(Level.five, "--prompt-level", "-pl", show_default=False,
                                           help="1-Text \n2-Text+description \n3-Text+format \n4-Text+Encoding \n5-Text+description+format"),
        random_instances: bool = typer.Option(False, "--random-instances", "-rn",
                                              help="Take random instances"),
) -> None:
    """
    Run selected testcases with Llama model.
    """
    assert app_options is not None

    with console.status("Loading data..."):
        data = load_data(app_options.dataset)
        problem_name = problem_name.replace(" ", "")
        problem_instances = [instance for instance in data
                             if instance["problem_name"].replace(" ", "") == problem_name]
        if random_instances:
            problem_instances = random.sample(problem_instances, quantity)
        else:
            problem_instances = problem_instances[:quantity]
        console.log("Data loaded!")
    console.log(f"Problem: {problem_name}; Instances: {quantity}")
    console.log(f"Testing model: {app_options.model}")

    results = test_model(
        app_options.model,
        problem_instances,
        app_options.server,
        app_options.ollama_key,
        prompt_level=prompt_level
    )
    console.log("Done with the model")
    save_results(results, file_path=app_options.output_file)
    console.print('---------------------------------Evaluate responses--------------------------------------')
    evaluate_model(app_options.model, results, app_options.stats_file)
    console.print('----------------------------------------Done---------------------------------------------')

def _llama_command_full_test_worker(prompt_level: Level, quantity: Optional[int]):
    """Worker for llama-full-test logic."""
    assert app_options is not None
    with console.status("Loading data..."):
        data = load_data(app_options.dataset)
        console.log("Data loaded!")
    console.log(f"Testing model: {app_options.model}")
    if quantity:
        data.sort(key=lambda i: i["problem_name"])
        grouped = groupby(data, key=lambda i: i["problem_name"])
        data = [inst for _, group in grouped for inst in list(group)[:quantity]]
    results = test_llama_dataset(
        app_options.model, app_options.server, app_options.ollama_key,
        data, prompt_level=prompt_level
    )
    console.log("Done with the model")
    save_results(results, file_path=app_options.output_file)
    console.print('---------------------------------Evaluate responses--------------------------------------')
    evaluate_model(app_options.model, results, app_options.stats_file)
    console.print('----------------------------------------Done---------------------------------------------')

@app.command(name="llama-full-test")
def llama_command_full_test(
    prompt_level: Level = typer.Option(Level.five, "--prompt-level", "-pl", show_default=False,
                                    help="Prompt level selection"),
    quantity: Optional[int] = typer.Option(None, "--quantity", "-q", help="Numero massimo di istanze per problema"),
):
    """Run for the entire dataset with Llama model."""
    assert app_options is not None
    if app_options.gpu_index is not None:
        run_with_energy_tracking(
            command_name="llama-full-test",
            log_file=app_options.energy_log_file,
            gpu_index=app_options.gpu_index,
            target_function=_llama_command_full_test_worker,
            prompt_level=prompt_level,
            quantity=quantity
        )
    else:
        _llama_command_full_test_worker(prompt_level, quantity)

def log_history(history):
    for item in history:
        console.log("[red]" + item[-2]["content"].replace('[', '\\['))
        console.log("[blue]" + item[-1]["content"].replace('[', '\\['))

def create_prompt(instance, prompt_level, output_format = None):
    # only text
    if prompt_level=='1':
        prompt = f"Extract the datalog facts from this text: \n ```{instance['text']}"
    # description and text
    elif prompt_level=='2':
        prompt = f"Given the following problem description between triple backtips: \n```{instance['description']}```\nExtract the datalog facts from this text: \n ```{instance['text']}```"
    # format and text
    elif prompt_level=='3':
        prompt = f"Given the following specification for the predicates format between triple backtips: \n```{instance['format']}```\nExtract the datalog facts from this text: \n ```{instance['text']}```"
    # only encoding
    elif prompt_level=='4':
        with open('data/dataset_encodings.json') as f:
            dataset_encodings = json.load(f)
        prompt = "Consider the following Datalog encoding for the problem, provided below within triple backticks: \n ```" + dataset_encodings[instance['problem_name']]['encoding'].strip() + "```\n"
        prompt += "Extract the datalog facts from this text: \n```" + instance['text'].strip() + "```\n"
        prompt += 'Output the result in datalog with no comments, no space between facts and only one fact per line.'
    else:
        prompt = instance['prompt']
    if output_format == "csv" or output_format == "csv_datalog":
        prompt += (
            "\nDo not include any explanation, comments, headers, or surrounding text."
            "\n\nOutput only the Datalog facts in CSV format."
            "Each row must represent exactly one fact."
            "The first column should be the predicate name, followed by one column per argument."
            "Start immediately. Do NOT explain or add any comments."

            "\nBEGIN CSV"
        )
    elif output_format == "json":
        prompt += (
            "\nRespond only with a JSON object in the following format:\n"
            "{\n"
            "  \"facts\": [\n"
            "    { \"predicate\": \"<predicate_name>\", \"arguments\": [<arg1>, <arg2>, ...] },\n"
            "    { \"predicate\": \"<predicate_name>\", \"arguments\": [<arg1>, <arg2>] }\n"
            "  ]\n"
            "}\n"
            "Each 'predicate' is a string, and each 'arguments' is an array of strings or numbers.\n"
            "Return only valid JSON. Do not include any explanations, comments, or extra text."
        )

    return prompt

def test_tool(tool, data: list, single_pass: bool=False):

    model = tool.llm.model
    partial_file = app_options.partial_file
    existing_results = load_partial_results(partial_file, model)
    done_ids = {
        json.dumps((inst["id"], inst["problem_name"], inst["text"])) 
        for inst in existing_results
    }

    progress = Progress(console=console)
    task_id = progress.add_task("Running...", completed=0, total=len(data))

    def live_grid(completed, table_data=None):
        progress.update(task_id, completed=completed)
        grid = Table.grid()
        grid.add_row(progress)
        if table_data is not None:
            grid.add_row(make_table("Stats", table_data))

        return grid

    results = existing_results.copy()
    
    with Live(console=console) as live:
        live.update(live_grid(0))
        for index, instance in enumerate(data):

            instance_id = json.dumps((instance["id"], instance["problem_name"], instance["text"]))
            if instance_id in done_ids:
                console.log(f"[yellow]Skipping already processed instance {index + 1}")
                continue

            query = instance["text"]
            try:
                created_facts, _, history, metadata = tool.natural_to_asp(query, single_pass=single_pass)
            except Exception as e:
                created_facts = ""
                history = []
                metadata = type("Metadata", (), {})()
                metadata.completion_tokens = 0
                metadata.prompt_tokens = 0
                metadata.total_tokens = 0
            if tool.config["knowledge_base"]:
                facts = []
                for fact in created_facts.split("\n"):
                    parsed = parse_fact_line(fact)
                    if not parsed:
                        console.log(f"Ignored atom (unrecognized format): {fact}")
                        continue
                    try:
                        facts.append(str(GroundAtom.parse(parsed)))
                    except Exception as e:
                        console.log(f"Ignored atom {parsed}: {e}")
                created_facts = Model.of_program(
                    Model.of_atoms(facts).as_facts,
                    tool.config["knowledge_base"]
                ).as_facts
            metadata_dict = {
                "completion_tokens": metadata.completion_tokens,
                "prompt_tokens": metadata.prompt_tokens,
                "total_tokens": metadata.total_tokens
            }
            metadata_str = json.dumps(metadata_dict)

            result = {**instance, **{model: created_facts}, **{"history": history}, **{"metadata": metadata_str}}
            results.append(result)

            with open(partial_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            if app_options.log_prompts:
                log_history(history)

            stats = evaluate_model(app_options.model, results, None)
            live.update(live_grid(index + 1, stats))

    return results

def test_problem(model, server, ollama_key, behavior_file, application_file, a1_prompt, data: list, single_pass: bool = False, output_format = None):

    llm_handler = llm.LLMHandler(
        model_name=model, 
        server_url=server, 
        api_key=ollama_key,
        output_format = output_format
    )
    
    tool = llm.LLMASP(
        config_file=application_file, 
        behavior_file=behavior_file, 
        llm=llm_handler,
        solver=None, 
        a1_prompt=a1_prompt
    ) 
    
    results = test_tool(tool, data, single_pass=single_pass)
    
    return results

def test_llmasp_dataset(model, server, ollama_key, behavior_file, application_files_folder, data: list, single_pass: bool=False, output_format=None, a1_prompt:bool=False):
    results = []
    data.sort(key=lambda i: i["problem_name"])
    grouped_problems = groupby(data, key=lambda p: p["problem_name"])

    for problem_name, instances in grouped_problems:
        instances = list(instances)
        console.log(f"Problem: {problem_name}; Instances: {len(instances)}")
        application_file = f"{application_files_folder}/{problem_name.replace(' ', '')}.yml"

        if not os.path.exists(application_file):
            console.log(f"File non trovato: {application_file}, salto il problema.")
            continue

        results.extend(
            test_problem(
                model,
                server,
                ollama_key,
                behavior_file,
                application_file,
                a1_prompt,
                instances,
                single_pass=single_pass,
                output_format=output_format,
            )
        )
    return results

def test_model(model_name, data, server_url: str="http://localhost:11434/v1", ollama_key="ollama", prompt_level=5):
    llm_handler = llm.LLMHandler(model_name, server_url=server_url, api_key=ollama_key)

    partial_file = app_options.partial_file
    existing_results = load_partial_results(partial_file, model_name)
    done_ids = {
        json.dumps((inst["id"], inst["problem_name"], inst["text"])) 
        for inst in existing_results
    }
    results = existing_results.copy()

    progress = Progress(console=console)
    task_id = progress.add_task("Running...", completed=0, total=len(data))

    def live_grid(completed, table_data=None):
        progress.update(task_id, completed=completed)
        grid = Table.grid()
        grid.add_row(progress)
        if table_data is not None:
            grid.add_row(make_table("Stats", table_data))

        return grid

    with Live(console=console) as live:
        live.update(live_grid(0))
        for index, instance in enumerate(data):

            instance_id = json.dumps((instance["id"], instance["problem_name"], instance["text"]))
            if instance_id in done_ids:
                console.log(f"[yellow]Skipping already processed instance {index + 1}")
                continue

            print("PROMPT LEVEL: ", prompt_level)
            prompt = create_prompt(instance, prompt_level)
            messages=[
            {
                "role": "system",
                "content": "You will be provided with unstructured data, and your task is to parse it into datalog facts."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
            completion, metadata = llm_handler.call(messages)

            metadata_dict = {
                "completion_tokens": metadata.completion_tokens,
                "prompt_tokens": metadata.prompt_tokens,
                "total_tokens": metadata.total_tokens
            }
            metadata_str = json.dumps(metadata_dict)
            
            result = {**instance, **{model_name: completion}, **{"history": messages}, **{"metadata": metadata_str}}
            results.append(result)

            with open(partial_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            live.update(live_grid(index + 1, evaluate_model(app_options.model, results, None)))

    return results

def test_llama_dataset(model, server, ollama_key, data, prompt_level):
    results = []
    data.sort(key=lambda i: i["problem_name"])
    grouped_problems = groupby(data, key=lambda p: p["problem_name"])
    for problem_name, instances in grouped_problems:
        instances = list(instances)
        console.log(f"Problem: {problem_name}; Instances: {len(instances)}")
        problem_name = problem_name.replace(" ", "")
        problem_results = test_model(model, instances, server, ollama_key, prompt_level)
        
        results.extend(problem_results)
    return results

@app.command(name="ollama-format-run-selected")
def ollama_format_run_selected(
    problem_name: str = typer.Option(..., "--problem-name", "-pn", help="Problem name"),
    quantity: int = typer.Option(..., "--quantity", "-q", help="Instances quantity"),
    prompt_level: Level = typer.Option(Level.five, "--prompt-level", "-pl", show_default=False,
                                       help="1-Text \n2-Text+description \n3-Text+format \n4-Text+Encoding \n5-Text+description+format"),
    random_instances: bool = typer.Option(False, "--random-instances", "-rn",
                                         help="Take random instances"),
    output_format: str = typer.Option(None, "--output-format", "-otf", 
                                      help="Output format (json, csv, etc.)"),
    temperature: float = typer.Option(0, "--temperature", "-t", help="Temperature for generation"),
) -> None:
   
    assert app_options is not None

    with console.status("Loading data..."):
        data = load_data(app_options.dataset)
        problem_name = problem_name.replace(" ", "")
        problem_instances = [instance for instance in data
                             if instance["problem_name"].replace(" ", "") == problem_name]
        if random_instances:
            problem_instances = random.sample(problem_instances, quantity)
        else:
            problem_instances = problem_instances[:quantity]
        console.log("Data loaded!")
    console.log(f"Problem: {problem_name}; Instances: {quantity}")
    console.log(f"Testing model: {app_options.model} with format: {output_format}")


    results = test_model_with_format(
        app_options.model,
        problem_instances,
        app_options.server,
        app_options.ollama_key,
        prompt_level=prompt_level,
        output_format=output_format,
        temperature=temperature
    )
    console.log("Done with the model")
    save_results(results, file_path=app_options.output_file)
    console.print('---------------------------------Evaluate responses--------------------------------------')
    evaluate_model(app_options.model, results, app_options.stats_file)
    console.print('----------------------------------------Done---------------------------------------------')

def _ollama_format_full_test_worker(prompt_level: Level, quantity: Optional[int], output_format: str, temperature: float):
    """Worker for ollama-format-full-test logic."""
    assert app_options is not None
    with console.status("Loading data..."):
        data = load_data(app_options.dataset)
    console.log("Data loaded!")
    console.log(f"Testing model: {app_options.model} with format: {output_format}")
    if quantity:
        data.sort(key=lambda i: i["problem_name"])
        grouped = groupby(data, key=lambda i: i["problem_name"])
        data = [inst for _, group in grouped for inst in list(group)[:quantity]]
    results = test_ollama_format_dataset(
        app_options.model, app_options.server, app_options.ollama_key, data,
        prompt_level=prompt_level, output_format=output_format, temperature=temperature
    )
    console.log("Done with the model")
    save_results(results, file_path=app_options.output_file)
    console.print('---------------------------------Evaluate responses--------------------------------------')
    evaluate_model(app_options.model, results, app_options.stats_file)
    console.print('----------------------------------------Done---------------------------------------------')

@app.command(name="ollama-format-full-test")
def ollama_format_full_test(
    prompt_level: Level = typer.Option(Level.five, "--prompt-level", "-pl", show_default=False,
                                      help="Prompt level selection"),
    quantity: Optional[int] = typer.Option(None, "--quantity", "-q", help="Numero massimo di istanze per problema"),
    output_format: str = typer.Option(None, "--output-format", "-otf", 
                                     help="Output format (json, csv, etc.)"),
    temperature: float = typer.Option(0, "--temperature", "-t", help="Temperature for generation"),
):
    """Run for the entire dataset with Ollama format mode."""
    assert app_options is not None
    if app_options.gpu_index is not None:
        run_with_energy_tracking(
            command_name="ollama-format-full-test",
            log_file=app_options.energy_log_file,
            gpu_index=app_options.gpu_index,
            target_function=_ollama_format_full_test_worker,
            prompt_level=prompt_level, quantity=quantity,
            output_format=output_format, temperature=temperature
        )
    else:
        _ollama_format_full_test_worker(
            prompt_level, quantity, output_format, temperature
        )

def test_model_with_format(model_name, data, server_url: str="http://127.0.0.1:11436", 
                          ollama_key="ollama", prompt_level=5, output_format=None, temperature=0):
    
    partial_file = app_options.partial_file
    existing_results = load_partial_results(partial_file, model_name)
    done_ids = {
        json.dumps((inst["id"], inst["problem_name"], inst["text"])) 
        for inst in existing_results
    }
    results = existing_results.copy()

    try:
        
        base_url = server_url
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
            
        health_url = f"{base_url}"
        
        health_check = requests.get(health_url, timeout=5)
    except Exception as e:
        console.log(f"[red]Server health check failed: {str(e)}")
        console.log("[yellow]Continuing anyway...")
    
    ollama_client = OllamaRequests(model_name, server_url=server_url, api_key=ollama_key)

    progress = Progress(console=console)
    task_id = progress.add_task("[cyan]Running...", completed=0, total=len(data))

    def live_grid(completed, table_data=None):
        progress.update(task_id, completed=completed)
        grid = Table.grid()
        grid.add_row(progress)
        if table_data is not None:
            grid.add_row(make_table("Stats", table_data))

        return grid

    with Live(console=console) as live:
        live.update(live_grid(0))
        for index, instance in enumerate(data):

            instance_id = json.dumps((instance["id"], instance["problem_name"], instance["text"]))
            if instance_id in done_ids:
                console.log(f"[yellow]Skipping already processed instance {index + 1}")
                continue
            
            prompt = create_prompt(instance, prompt_level, output_format)
            
            system_prefix = "You will be provided with unstructured data, and your task is to parse it into datalog facts.\n\n"
            full_prompt = system_prefix + prompt
            
            completion, metadata = ollama_client.call(
                full_prompt, 
                format=output_format,
                temperature=temperature
            )
            
            history = [
                {"role": "system", "content": system_prefix.strip()},
                {"role": "user", "content": prompt}
            ]

            result = {**instance, **{model_name: completion}, **{"history": history}, **{"metadata": metadata}}
            results.append(result)

            with open(partial_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            stats = evaluate_model(app_options.model, results, None)
            live.update(live_grid(index + 1, stats))

    return results

def test_ollama_format_dataset(model, server, ollama_key, data, prompt_level, output_format=None, temperature=0):

    results = []
    data.sort(key=lambda i: i["problem_name"])
    grouped_problems = groupby(data, key=lambda p: p["problem_name"])
    for problem_name, instances in grouped_problems:
        instances = list(instances)
        console.log(f"Problem: {problem_name}; Instances: {len(instances)}")
        problem_name = problem_name.replace(" ", "")
        problem_results = test_model_with_format(
            model, 
            instances, 
            server, 
            ollama_key, 
            prompt_level, 
            output_format=output_format,
            temperature=temperature
        )
        results.extend(problem_results)
    return results


