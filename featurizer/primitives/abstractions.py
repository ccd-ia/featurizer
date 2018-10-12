# coding: utf-8


class Entity:
    def __init__(self, alias, table, id, spatial=None, temporal=None, variables=None):
        self.alias = alias
        self.id = Id(name=id, entity=self)
        self.table = table

        self.features = [ Variable(name=var, type=description['type'], entity=self) for var, description in variables.items() ]
        self.relationships = []

        self.spatial = spatial
        self.temporal = temporal

    def __repr__(self):
        return f"Entity({self.alias})"

    def info(self):
        return f"""

        {self.alias.capitalize()}(table = {self.table})

            Variables:
               {self.variables}

        """

    def add_features(self, features):
        for feature in features:
            if feature not in self.features:
                self.features.append(feature)

    def add_relationship(self, relationship):
        self.relationships.append(relationship)


class Relationship:
    def __init__(self, parent, child):

        self.parent = parent['entity']
        self.parent_key = parent['key']
        self.child = child['entity']
        self.child_key = child['key']

    def __repr__(self):
        return f"""{self.parent}.{self.parent_key} -> {self.child}.{self.child_key}"""

    def __eq__(self, other):
        return self.parent == other.parent and \
            self.parent_key == other.parent_key and \
            self.child == other.child and \
            self.child_key == other.child_key


class Feature:
    """ Base class for features """
    def __init__(self, name, type, definition=None, entity=None, parents=None, intervals=None, specials=None, stack_depth=0):
        self.name = name
        self.type = type
        self.definition = definition if definition else f"{entity.alias}.{name}"
        self.stack_depth = stack_depth
        self.entity = entity
        self.parents = parents
        self.intervals = intervals
        self.specials = specials

    def __repr__(self):
        return f"""Feature({self.name.replace('"', '')})"""

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name) ^ hash(self.type) ^ hash(self.definition) ^ \
            hash(self.from_entity) ^ \
            hash((self.name, self.type, self.definition, self.from_entity))

class Variable(Feature):
    """ Represents a column in a table. """
    def __init__(self, name, type, entity):
        super().__init__(name=name, definition=name, type=type, entity=entity, stack_depth=-1)


class Id(Feature):
    """ Represents an entity id """
    def __init__(self, name, entity):
        super().__init__(name=name, definition=name, type='index', entity=entity)
