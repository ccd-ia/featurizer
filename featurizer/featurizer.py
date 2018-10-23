# coding: utf-8

from functools import partialmethod

import records
import yaml

from .primitives import Entity, Relationship
from .primitives import Aggregator, Transformer
from .primitives import ERGraph

from .primitives.aggregations import *
from .primitives.transformations import *

AGGREGATIONS = {
#    'sum': sum,
    'count': count,
    'mean': mean,
    # 'median': median,
    # 'mode': mode
}

TRANSFORMATIONS = {
    # 'first': first,
    # 'last': last,
    # 'previous': previous,
    # 'day': day,
#    'month': month,
    # 'dow': dow,
    # 'hourly_binning': hourly_binning,
    # 'daily_binning': daily_binning,
#    'isnull': isnull
    #'ln': ln
    'identity': identity
}

class Featurizer:
    """
    PostgreSQL implementation of the
    DFS algorithm (adapted for temporal data sets)
    """

    def __init__(self, config_file):
        with open(config_file) as f:
            config = yaml.load(f)

        self.graph = ERGraph(config['entities'], config['relationships'])

        self.target = self.get_entity(config['target'])

        self.intervals = config['intervals']

        self.ctes = []

        self.path = []

        self.features = {e.alias: set(e.features) for e in self.entities}

        self.joins = {e.alias: [] for e in self.entities}

        self.make_features()

    @property
    def entities(self):
        return self.graph.entities.values()

    @property
    def relationships(self):
        return self.graph.relationships

    def get_entity(self, alias):
        return self.graph.entities.get(alias, None)

    @property
    def query(self):
        return f"""
        select aod.as_of_date, t.*
        from as_of_dates as aod
        cross join lateral (

        with

        {','.join(self.ctes)}

        select * from {self.target.alias}_transform
        ) as t

        order by aod.as_of_date
        """

    def make_features(self):
        return self.build_features(self.target)

    def build_features(self, target_entity, i=0):
        print("\t"*i, f"build_features({target_entity.alias})")

        i = i+1

        if target_entity not in self.path:
            self.path.append(target_entity)

        self.get_direct_features(target_entity, i)

        self.get_backward_features(target_entity, i)

        self.build_transformations(target_entity)


    def get_direct_features(self, target, i):
        forward_relationships = self.graph.get_forward_relationships(target)
        for fr in forward_relationships:
            e = fr.parent
            if e in self.path:
                continue
            self.build_features(e, i)
            self.build_direct(target, e, fr)

    def get_backward_features(self, target, i):
        backward_relationships = self.graph.get_backward_relationships(target)
        for br in backward_relationships:
            e = br.child
            if e in self.path:
                continue
            self.build_features(e, i)
            self.build_aggregations(target, e, br)

    def build_aggregations(self, target, entity, br):
        print(br)
        aggs = []

        for feature in self.features[entity.alias]:
            for agg_name, aggregator in AGGREGATIONS.items():
                new_feature = aggregator(target, entity, feature)
                if new_feature:
                    aggs.append(new_feature)
                for interval in self.intervals:
                    print(f"{feature}")
                    print(f"Interval: {interval}")
                    if feature.entity.temporal_ix is None:
                        print(f"Entity: {feature.entity} doesn't have event dates. Skipping")
                        break

                    new_feature = aggregator(target,
                                             entity,
                                             feature,
                                             interval=interval)
                    if new_feature:
                        aggs.append(new_feature)

        aggs = set(aggs)

        self.features[entity.alias].update(aggs)
        self.features[target.alias].update(aggs)  # synthetize

        self.build_aggregations_cte(target, entity, br, aggs)


    def build_direct(self, target, entity, fr):
        print(fr)
        directs = []
        for feature in self.features[entity.alias]:
            directs.append(feature)

        directs = set(directs)

        self.features[target.alias].update(directs)

        self.build_direct_cte(target, entity, fr, directs)


    def build_transformations(self, target):

        self.build_synthetize_cte(target)

        trans = []

        for feature in self.features[target.alias]:
            if feature.type != 'index':
                for trans_name, transformer in TRANSFORMATIONS.items():
                    new_feature = transformer(target, feature)
                    if new_feature:
                        trans.append(new_feature)
            else:
                trans.append(feature)

        trans = set(trans)

        self.features[target.alias].update(trans)

        self.build_transform_cte(target, trans)

    def build_aggregations_cte(self, target, entity, br, aggs):
        cte_name=f"{entity.alias}_aggs_for_{target.alias}"
        join_statement = f" {cte_name} on {cte_name}.{br.child_key} = {br.parent.table}.{br.parent_key} "

        fts = [agg.query for agg in aggs if agg.type not in ['key']]
        where_clause = f"where aod.as_of_date >= {entity.temporal_ix.name}" if entity.temporal_ix else ''

        cte_query = f"""
        -- Aggregate for {target.alias}
        {cte_name} as (
        select
        {entity.alias}_transform.{br.parent_key},
        {','.join(fts)}
        from {entity.alias}_transform
        {where_clause}
        group by {br.parent_key}
        )
        """
        self.joins[target.alias].append( join_statement )
        self.ctes.append(cte_query)

    def build_direct_cte(self, target, entity, fr, directs):
        cte_name = f"{entity.alias}_direct_transfers_for_{target.alias}"

        cte_query = f"""
        -- direct features for {target.alias}
        {cte_name} as (
        select
        {entity.id.name},
        {','.join([direct.name for direct in directs if direct.type not in ['index', 'key']])}
        from {entity.alias}_transform
        )
        """
        join_statement = f" {cte_name} on {cte_name}.{fr.child_key} = {fr.child.table}.{fr.child_key} "
        self.joins[target.alias].append(join_statement)
        self.ctes.append(cte_query)

    def build_synthetize_cte(self, target):

        cte_table = f"{target.alias}_synth"

        ixs = [f"{target.table}.{ix.name}" for ix in target.indexes]
        keys = [f"{target.table}.{key.name}" for key in target.keys]
        fts = [ft.name for ft in self.features[target.alias] if ft.type not in ['index', 'key']]

        cte_query = f"""
        -- sythetize aggregations and direct features for {target.alias}
        {cte_table} as (
        select
        {', '.join(ixs + keys + fts)}
        from {target.table}
        {' left join ' if self.joins[target.alias] else '' }
        {' left join '.join([ join_statement for join_statement in self.joins[target.alias]])}
        )
        """

        self.ctes.append(cte_query)

    def build_transform_cte(self, target, trans):
        cte_table = f"{target.alias}_transform"

        ixs = [f"{ix.name}" for ix in target.indexes]
        keys = [f"{key.name}" for key in target.keys]
        fts = [ft.query for ft in trans if ft.type not in ['index', 'key']]

        cte_query = f"""
        -- transform {target.alias}
        {cte_table} as (
        select
        {', '.join(ixs + keys + fts)}
        from {target.alias}_synth
        )
        """

        self.ctes.append(cte_query)

    def to_dataframe(self):
        db = records.Database()
        rows = db.query(self.query)
        df = rows.export('df')
        df = df.set_index(['as_of_date', self.target.id.name], inplace=False)
        return df
