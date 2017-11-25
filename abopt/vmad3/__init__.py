from .registry import Registry

default_registry = Registry()

def register(primitive):
    default_registry.register(primitive, primitive.get_type_patterns)
    return primitive
