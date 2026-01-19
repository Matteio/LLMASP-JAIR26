import yaml
import re
import json
from typing import Optional

from .abstract_llmasp import AbstractLLMASP
from .llm_handler import LLMHandler

class AccumulatedMetadata:
    def __init__(self):
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
        self.individual_calls = []
    
    def add_metadata(self, meta):
        self.completion_tokens += meta.completion_tokens
        self.prompt_tokens += meta.prompt_tokens
        self.total_tokens += meta.total_tokens
        self.individual_calls.append({
            "completion_tokens": meta.completion_tokens,
            "prompt_tokens": meta.prompt_tokens,
            "total_tokens": meta.total_tokens
        })

class LLMASP(AbstractLLMASP):
    
    def __init__(self, config_file: str, behavior_file: str, llm: LLMHandler, solver, a1_prompt: bool = False):
        super().__init__(config_file, behavior_file, llm, solver, a1_prompt)
        try:
            self.a1_prompt = a1_prompt
            db_file = self.load_file(self.config["database"])
            self.database = db_file["database"]
        except:
            self.database = ""

    def __get_atom_name(self, atom: str):
        return atom.split("(")[0]

    def __prompt(self, role: str, content: str):
        return { "role": role, "content": content }

    def __create_queries(self, user_input: str, single_pass: bool = False, output_format: Optional[str] = None):
        queries = []
        
        init_template = self.behavior["preprocessing"]["init"]
        context_template = self.behavior["preprocessing"]["context"]
        mapping_template = self.behavior["preprocessing"]["mapping"]
        
        _, application_context_text = self.__get_property(self.config["preprocessing"], "_")
        processed_context_prompt = context_template.replace("{context}", application_context_text)
        
        user_message_base_template = re.sub(r"\{input\}", user_input, mapping_template)

        additional_instructions = ""

        if output_format == "csv" or output_format == "csv_datalog":
            additional_instructions += (
                "\nDo not include any explanation, comments, headers, or surrounding text."
                "\n\nOutput only the Datalog facts extracted from the INPUT in CSV format. "
                "Each row must represent exactly one fact. "
                "The first column should be the predicate name, followed by one column per argument. "
            )

        elif output_format == "json":
            additional_instructions += (
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


        if single_pass:
            formats, instructions_list = zip(*[(key, value) for query_config in self.config["preprocessing"]
                                         for key, value in query_config.items() if key != "_"])
            application_mapping = re.sub(r"\{instructions\}", "".join(instructions_list), user_message_base_template)
            application_mapping = re.sub(r"\{atom\}", " ".join(formats), application_mapping)
            application_mapping += additional_instructions


            if self.a1_prompt:
                    queries.append([
                    self.__prompt("system", init_template),
                    self.__prompt("user", application_mapping_for_query)
                ])
            else:
                queries.append([
                    self.__prompt("system", init_template),
                    self.__prompt("system", processed_context_prompt),
                    self.__prompt("user", application_mapping_for_query)
                ])

        else:  
            for query_config in self.config["preprocessing"]:
                
                if "_" in query_config:
                    continue

                key_atom, value_instructions = list(query_config.items())[0]

                if output_format == "csv": 
                    match = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)", key_atom)
                    if match:
                        pred_name = match.group(1)
                        args = match.group(2).replace(" ", "").split(",")
                        key_atom_csv = ",".join([pred_name] + args)
                    else:
                        key_atom_csv = key_atom
                else:
                    key_atom_csv = key_atom

                application_mapping_for_query = re.sub(r"\{instructions\}", value_instructions, user_message_base_template)
                application_mapping_for_query = re.sub(r"\{atom\}", key_atom_csv, application_mapping_for_query)
                application_mapping_for_query += additional_instructions


                if self.a1_prompt:
                    queries.append([
                    self.__prompt("system", init_template),
                    self.__prompt("user", application_mapping_for_query)
                ])
                else:
                    queries.append([
                        self.__prompt("system", init_template),
                        self.__prompt("system", processed_context_prompt),
                        self.__prompt("user", application_mapping_for_query)
                    ])

        return queries

    

    def __get_property(self, properties, key, is_fact=False):
        if is_fact:
            property = list(filter(lambda x: self.__get_atom_name(next(iter(x))) == key, properties))[0]        
        else:
            property = list(filter(lambda x: next(iter(x)) == key, properties))[0]
        property_key = next(iter(property))
        property_value = list(property.values())[0]
        return property_key, property_value
    
    def load_file(self, config_file: str):
        return yaml.load(open(config_file, "r"), Loader=yaml.Loader)
    
    def asp_to_natural(self, facts:list, history: list, use_history: bool = True):

        def group_by_fact(facts: list) -> dict:
            grouped = {}
            for f in facts:
                name = self.__get_atom_name(f)
                grouped.setdefault(name, []).append(f)
            return grouped
        grouped_facts = group_by_fact(facts)

        responses = []
        if use_history == True:
            queries = [x for v in history for x in v]
        else:
            queries = []
        context = self.behavior["postprocessing"]["context"]
        
        _, application_context = self.__get_property(self.config["postprocessing"], "_")
        application_context = re.sub(r"\{context\}", application_context, context)
        final_response = self.behavior["postprocessing"]["summarize"]
        for fact_name in grouped_facts:
            fact_translation = self.behavior["postprocessing"]["mapping"]
            f_translation_key, f_translation_value = self.__get_property(self.config["postprocessing"], fact_name, is_fact=True)
            fact_translation = re.sub(r"\{atom\}", f_translation_key, fact_translation)
            fact_translation = re.sub(r"\{intructions\}", f_translation_value, fact_translation)
            fact_translation = re.sub(r"\{facts\}", "\n".join(grouped_facts[fact_name]), fact_translation)
            res = self.llm.call([*queries, *[
                    self.__prompt("system", self.behavior["postprocessing"]["init"]),
                    self.__prompt("system", application_context),
                    self.__prompt("user", fact_translation),
                ]])
            responses.append(res)
        
        final_response = re.sub(r"\{responses\}", "\n".join(responses), final_response)
        return self.llm.call([self.__prompt("system", application_context), self.__prompt("user", final_response)])

    def natural_to_asp(self, user_input:str, single_pass: bool = False, max_tokens:Optional[int] = None):
        
        output_fmt_from_handler = None
        if hasattr(self.llm, 'output_format') and self.llm.output_format:
            output_fmt_from_handler = self.llm.output_format.lower() 
        
        queries = self.__create_queries(user_input, single_pass=single_pass, output_format=output_fmt_from_handler)
        created_facts = ""

        accumulated_meta = AccumulatedMetadata()

        for q in queries:
            facts, meta = self.llm.call(q, max_tokens=max_tokens)
            accumulated_meta.add_metadata(meta)

            if output_fmt_from_handler == "json":
                try:
                    print("[DEBUG] Raw 'facts' string from LLM:", repr(facts))

                    decoded = json.loads(facts)
                    print("[DEBUG] After first json.loads:", decoded)

                    if isinstance(decoded, str):  
                        decoded = json.loads(decoded)
                        print("[DEBUG] After second json.loads (nested JSON):", decoded)

                    if isinstance(decoded, dict) and "facts" in decoded:
                        print("[DEBUG] 'facts' key found, processing list...")
                        facts_list = [
                            f"{item['predicate']}({','.join(map(str, item['arguments']))})."
                            for item in decoded["facts"]
                            if "predicate" in item and "arguments" in item
                        ]
                        print("[DEBUG] Parsed facts list:", facts_list)
                        facts = "\n".join(facts_list)
                    else:
                        print("[WARN] JSON object doesn't contain expected 'facts' key.")
                        facts = ""
                except Exception as e:
                    print(f"[ERROR] Errore nel parsing JSON: {e}")
                    facts = ""
                    
            elif output_fmt_from_handler == "csv":
                facts = "\n".join(facts.strip().splitlines())

            else: 
                facts = re.findall(r"\b[a-zA-Z][\w_]*\([^)]*\)", facts)
                facts = [f'{f}.' for f in facts]
                facts = "\n".join(facts)

            created_facts = f"{created_facts}\n{facts}"

            q.append(self.__prompt("assistant", facts))
        asp_input = f"{created_facts}\n{self.database}\n{self.config["knowledge_base"]}"
        return created_facts, asp_input, queries, accumulated_meta

    def run(self, user_input: str, single_pass: bool = False, use_history: bool = False, verbose: int = 0):
        try:
            logs = []
            output = ""
            logs.append(f"input: {user_input}")
            created_facts, asp_input, history = self.natural_to_asp(user_input, single_pass=single_pass)
            logs.append(f"extracted facts: {created_facts}")

            result, _, _ = self.solver.solve(asp_input)
            if (len(result) == 0):
                logs.extend(["answer set: not found", "out: not found"])
            else:
                logs.append(f"answer set: {result}")
                response = self.asp_to_natural(result, history, use_history=use_history)
                logs.append(f"output: {response}")
                output = response
            if (verbose == 1):
                print("\n\n".join(logs))
            return output
        except:
            print("Error: Generic error.")
