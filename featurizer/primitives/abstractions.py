# coding: utf-8


class ERGraph:
    def __init__(self, entities, relationships):
        self.entities = {e['alias']: Entity(**e) for e in entities}

        if relationships:
            self.relationships = [
                Relationship(
                    parent = self.entities[r['parent']['entity']],
                    child = self.entities[r['child']['entity']],
                    parent_key = r['parent']['key'],
                    child_key = r['child']['key']
                )
                for r in relationships
            ]
        else:
            self.relationships = {}

        for r in self.relationships:
            self.entities[r.child.alias].add_key(Key(name=r.child_key, entity=r.child))

    def get_backward_entities(self, entity):
        return {r.child for r in self.relationships if r.parent == entity}

    def get_forward_entities(self, entity):
        return {r.parent for r in self.relationships if r.child == entity}

    def get_backward_relationships(self, entity):
        return {r for r in self.relationships if r.parent == entity}

    def get_forward_relationships(self, entity):
        return {r for r in self.relationships if r.child == entity}

class Entity:
    def __init__(self, alias, table, id, spatial=None, temporal=None, variables=None):
        self.alias = alias
        self.id = Id(name=id, entity=self) if id else None
        self.table = table

        self.keys = []

        self.features = [ Variable(name=var, type=description['type'], entity=self) for var, description in variables.items() ]

        self.features = self.features + ([self.id] if self.id else [])

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

    def add_key(self, key):
        if key not in self.keys:
            self.keys.append(key)

    def add_features(self, features):
        for feature in features:
            if feature not in self.features:
                self.features.append(feature)

    def add_relationship(self, relationship):
        if self.alias in relationship:
            if self.alias == relationship.parent:
                self.relationships['backward'].append(relationship)
            else:
                self.relationships['forward'].append(relationship)

    def get_relationship(self, other, type):
        relationship = None
        for r in self.relationships[type]:
            if other in r:
                relationship = r
                break

        return relationship

class Relationship:
    def __init__(self, parent, child, parent_key, child_key):

        self.parent = parent
        self.parent_key = parent_key
        self.child = child
        self.child_key = child_key

    def __repr__(self):
        return f"""{self.parent}.{self.parent_key} -> {self.child}.{self.child_key}"""

    def __eq__(self, other):
        return self.parent == other.parent and \
            self.parent_key == other.parent_key and \
            self.child == other.child and \
            self.child_key == other.child_key

    def __hash__(self):
        return hash(self.parent.alias) and hash(self.parent_key) \
            and hash(self.child.alias) and hash(self.child_key)

    def __contains__(self, entity):
        if entity in [self.parent, self.child]:
            return True

    def is_backward(self, e1, e2):
        return e1 == self.parent and e2 == self.child

    def is_forward(self, e1, e2):
        return e1 == self.child and e2 == self.parent


class Feature:
    """ Base class for features """
    def __init__(
            self, name, type, definition=None, entity=None, parents=None,
            intervals=None, specials=None, sort=None, description='a feature',
            stack_depth=0
    ):
        self.name = name
        self.type = type
        self.definition = definition
        self.stack_depth = stack_depth
        self.entity = entity
        self.parents = parents  ## Which are they parent variables
        self.intervals = intervals or [] ## Do we care about some past time intervals?
        self.specials = specials or []  ## Do we care about specific values?
        self.sort = sort ## Sort by...
        self.description = description

    def __repr__(self):
        return f"""Feature({self.name.replace('"', '')})"""

    def __eq__(self, other):
        return self.name == other.name

    def __neq__(self, other):
        return self.name != other.name

    def __hash__(self):
        return hash(self.name) ^ hash(self.type) ^ hash(self.definition) ^ \
            hash((self.name, self.type, self.definition))

    @property
    def query(self):
        return f"""{self.definition} as "{str.replace(self.name, '"', '')}" """


class Variable(Feature):
    """ Represents a column in a table. """
    def __init__(self, name, type, entity):
        super().__init__(name=name, definition=name, type=type, entity=entity, stack_depth=0)


class Id(Feature):
    """ Represents an entity id """
    def __init__(self, name, entity):
        super().__init__(name=name, definition=name, type='index', entity=entity)


class Key(Feature):
    """ Represents a reference to another table """
    def __init__(self, name, entity):
        super().__init__(name=name, definition=name, type='key', entity=entity)
