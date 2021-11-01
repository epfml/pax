import os

class Registry:
    def __init__(self):
        self.registry = {}
    
    def register(self, identifier, implementation):
        self.registry[identifier] = implementation
    
    def __call__(self, identifier):
        if identifier in self.registry:
            return self.registry[identifier]
        else:
            raise ValueError(f"Could not find {identifier}")
    
    def __setitem__(self, identifier, implementation):
        self.register(identifier, implementation)
    
    def __getitem__(self, identifier):
        if identifier in self.registry:
            return self(identifier)
        else:
            super()(identifier)
    
    def list(self):
        return list(self.registry.keys())

task = Registry()
model = Registry()
dataset = Registry()
config = Registry()

config["data_root"] = os.path.join(os.getenv("HOME"), "pax_data")
