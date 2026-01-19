
from abc import ABC, abstractmethod

class AbstractLLMASP(ABC):

    def __init__(self, config_file:str, behavior_file:str, llm, solver, a1_prompt):
        self.config = self.load_file(config_file)
        self.behavior = self.load_file(behavior_file)
        self.llm = llm
        self.solver = solver
        self.a1_prompt = a1_prompt

    @abstractmethod
    def load_file(path:str):
        pass

    @abstractmethod
    def asp_to_natural(self):
        pass
    
    @abstractmethod
    def natural_to_asp(self):
        pass
    
    @abstractmethod
    def run(self, input, verbose):
        pass