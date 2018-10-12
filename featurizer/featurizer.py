# coding: utf-8

from functools import partialmethod

import records
import yaml

from .primitives import Entity, Relationship
from .primitives import Aggregator, Transformer

class Featurizer:
    """
    PostgreSQL implementation of the
    DFS algorithm (adapted for temporal data sets)
    """

    def __init__(self, config_file):
        with open(config_file) as f:
            config = yaml.load(f)

        self.entities = {e['alias']: Entity(**e) for e in config['entities']}

        self.relationships = [ Relationship(**r) for r in config['relationships'] ]

        self.ctes = []

        self.query = []

        self.features = { e.alias: {'agg':[], 'trans':[], 'direct': []}  for e in self.entities.values() }

        self._create_graph()

    def get_entity(self, alias):
        return self.entities.get(alias, None)

    def _create_graph(self):
        graph = {}
        for e in self.entities:
            graph[e] = []

        for r in self.relationships:
            graph[r.parent].append(r)

        # Pruning
        self.graph = {node:edges for node, edges in graph.items() if edges}

    def create_query(self, target):

        def get_features_statement(entity):
            return ','.join([f'{feature.definition} as "{feature.name}"' for feature in entity.features])

        def get_join_statement(target):
            return ' '.join([f"inner join {r.child}_cte as {r.child} on {r.parent}.{r.key} = {r.child}.{r.key}" for r in self.relationships if r.parent == target])

        ctes = []

        for e in self.entities.values():
            if e == target:
                continue

            cte = f"""
            {e.alias}_cte as (
                select
                {get_features_statement(e)}
                from
                {e.table} as {e.alias}
            )
            """

            ctes.append(cte)

        query = f"""

        with
        {','.join(ctes)}

        select
        {get_features_statement(target)}
        from
        {target.table}
        {get_join_statement(target)}
        """

        return  query

    def build_features(self, target_entity, visited_entities=None, i=0):
        print("\t"*i, f"build_features({target_entity.alias})")

        i = i+1

        if visited_entities is None:
            visited_entities = set()

        visited_entities.add(target_entity.alias)
        print("\t"*i, visited_entities)
        backward_entities = self.get_backward(target_entity)
        print("\t"*i, f"E_B({target_entity.alias}): {backward_entities}")
        forward_entities = self.get_forward(target_entity)
        print("\t"*i, f"E_F({target_entity.alias}): {forward_entities}")

        for e in forward_entities:
            print("\t"*i,e)
            if e.alias in visited_entities:
                print("\t"*i, f"Ignoring {e}")
                continue
            self.build_features(e, visited_entities, i)
            print("\t"*i, f"Agregando features directas a {target_entity}")
            self.features[target_entity.alias]['direct'].extend(self.build_direct(target_entity, e))

        for e in backward_entities:
            if e.alias in visited_entities:
                print("\t"*i, f"Ignoring {e}")
                continue
            self.build_features(e, visited_entities, i)
            print("\t"*i, f"Agregando features agregadas a {e}")
            self.features[e.alias]['agg'].extend(self.build_aggregations(target_entity, e))

        print("\t"*i, f"Agregando features transformadas a {target_entity}")
        self.features[target_entity.alias]['trans'].extend(self.build_transformations(target_entity))

    def get_backward(self, entity):
        return set([ self.entities[r.child] for r in self.relationships if r.parent == entity.alias])

    def get_forward(self, entity):
        return set([ self.entities[r.parent] for r in self.relationships if r.child == entity.alias])

    def build_aggregations(self, target, entity):
        # aggregations
        aggs = []
        for feature in entity.features:
            if feature.type == 'numeric':
                aggs.append(self.sum(target, entity, variable=feature))
            elif feature.type == 'categorical':
                aggs.append(self.count(target, entity, variable=feature))
            else:
                aggs.append(feature)
        return aggs

    def build_direct(self, target, entity):
        direct = []
        for feature in entity.features:
            direct.append(feature)
        return direct

    def build_transformations(self, target):
        trans = []
        for feature in target.features:
            if feature.type == 'numeric':
                trans.append(self.abs(target, variable=feature))
            elif feature.type == 'date':
                trans.append(self.dow(target, date_var=feature))
            elif feature.type == 'text':
                trans.append(self.num_chars(target, text_var=feature))
            else:
                trans.append(feature)
        return trans



# with open('featureizer.yaml') as f:
#     FEATURES_CONFIG = yaml.load(f)

# with open('aggregates.sql') as f:
#     AGGREGATE_TEMPLATE = f.read()


# def get_queries(as_of_dates):
#     queries = Template(
#         AGGREGATE_TEMPLATE
#     ).render(
#         features=FEATURES_CONFIG,
#         as_of_dates=as_of_dates
#     )

#     print(queries)

#     return queries


# def run_query():
#     print("Query executed")


# def query_to_pandas(as_of_dates):
#     queries = get_queries(as_of_dates)

#     return pd.DataFrame()


# def query_to_table(as_of_dates):
#     queries = get_queries(as_of_dates)

#     print("Table created")
