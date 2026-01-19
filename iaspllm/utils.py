import json
import os
import re
import csv
from dumbo_utils.console import console


def load_data(path):
    with open(path) as f:
        data = json.load(f)
    return data

def process_string(s: str) -> list[str]:
    s = s.strip()

    try:
        intermediate = json.loads(s)
        data = json.loads(intermediate)

        if "facts" in data:

            facts_list = data["facts"]
            facts = []
            for fact in facts_list:
                if isinstance(fact, dict):
                    pred = fact.get("predicate")
                    args = fact.get("arguments", [])
                    args_str = ",".join(str(a) for a in args)
                    facts.append(f"{pred}({args_str})")
                else:
                    console.log(f"[red]Ignored non-dict fact: {fact}")
            return facts

    except json.JSONDecodeError:
        pass

    if '(' in s and ')' in s:
        facts = re.findall(r"\b[a-zA-Z_][\w_]*\([^)]*\)", s)
        return [fact.replace(" ", "") for fact in facts]
    
    lines = s.splitlines()
    facts = []
    for line in lines:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2:  # almeno pred e un arg
            pred = parts[0]
            args = ",".join(parts[1:])
            facts.append(f"{pred}({args})")
    return facts

def save_results(results, file_path='llama_results.json'):
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    unique = {}
    for r in results:
        key = json.dumps((r["id"], r["problem_name"]))
        unique[key] = r


    try:
        with open(file_path, 'w', encoding='utf-8') as outfile:
            json.dump(list(unique.values()), outfile, indent=4, ensure_ascii=False)
        print(f"success saved: {file_path}")
    except Exception as e:
        print(f"error saving: {e}")


def parse_fact_line(line):
    line = line.strip()
    if not line:
        return None  

    if '(' in line and ')' in line:
        if line.endswith('.'):
            return line[:-1]  
        else:
            return line

    if "," in line:
        parts = line.split(",")
        if len(parts) >= 2:
            pred = parts[0]
            args = ",".join(parts[1:])
            return f"{pred}({args})"  
    return None

def evaluate_predictions(y_pred, y_true):
    y_pred, y_true = set(y_pred), set(y_true)

    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    true_positives = len(y_pred.intersection(y_true))
    false_positives = len(y_pred - y_true)
    false_negatives = len(y_true - y_pred)

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = true_positives / len(y_pred) if len(y_pred) > 0 else 0

    # Output results
    results = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    return results


def evaluate_model(model_name, data, stats_file):
    s = []
    overall = {'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'F1 Score': 0.0}
    unique_problems = set([problem['problem_name'] for problem in data])
    problem_metrics = {}
    for problem in unique_problems:
        problem_metrics[problem] = {'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'F1 Score': 0.0, 'Count': 0}
    
    for instance in data:
        y_true, y_pred = instance["output"], instance[model_name]
        y_true = process_string(y_true)
        y_pred = process_string(y_pred)
        instance_metrics = evaluate_predictions(y_pred, y_true)
        
        for metric, value in instance_metrics.items():
            overall[metric] += value
            problem_metrics[instance['problem_name']][metric] += value
        problem_metrics[instance['problem_name']]['Count'] += 1

    table = [['Problem Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score']]
    for problem, metrics in problem_metrics.items():
        s.append([problem] + [f'{value/metrics["Count"]:.2f}' for metric, value in metrics.items() if metric != 'Count'])

    s = sorted(s, key=lambda x: x[0])
    s.append(["Overall Results"] + [f'{value/len(data):.2f}' if len(data) > 0 else 0 for _, value in overall.items()])
    table.extend(s)

    if stats_file is not None:
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        with open(stats_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(table)
    else:
        return table

def load_partial_results(jsonl_path, model_name):
    if not os.path.exists(jsonl_path):
        return []
    
    results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get(model_name, "").strip():
                results.append(data)
    return results