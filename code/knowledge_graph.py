# knowledge_graph.py

# knowledge_graph.py
kg = {
    "New York": {
        "location": {"Park Avenue", "Times Square", "Wall Street"}
    },
    "Los Angeles": {
        "location": {"Sunset Boulevard", "Hollywood Blvd", "Park Avenue"}
    }
}
knowledge_graph = {
    "NYC": {
        "state": ["New York"],
        "population": ["8M"],
        "type": ["city"]
    },
    "MIT": {
        "state": ["Massachusetts"],
        "type": ["university"]
    }
}

knowledge_triples = []
def is_valid_entity_attribute(entity, attr_name, attr_value):
    return attr_value in kg.get(entity, {}).get(attr_name, set())


def get_related_entities(entity):
    related = set()
    for h, r, t in knowledge_triples:
        if entity.lower() == h.lower():
            related.add(t.lower())
        elif entity.lower() == t.lower():
            related.add(h.lower())
    return related
