import re

from helpers.sql import SQL
from bs4 import BeautifulSoup
from objs.Relation import Relation, Relations


class Request:

    fillables = {
        'request': ['id', 'description', 'subject', 'solution', 'timeConsumption', 'receivedDate', 'solutionDate', 'deadline', 'priority'],
        'request_tasktype_simple': ['id', 'description', 'subject', 'tasktype']
    }

    def __init__(self, request_id):
        self.request_id = request_id
        self.relations = None
        self.relations_history = None
        self.subject = ''
        self.description = ''
        self.timeConsumption = ''
        self.receivedDate = ''
        self.solutionDate = ''

    def pretty(self):
        obj = {
            'id': self.request_id,
            'subject': self.subject,
            'description': self.description,
            'timeConsumption': self.timeConsumption,
            'receivedDate': self.receivedDate,
            'solutionDate': self.solutionDate
        }
        if self.relations is not None:
            obj['relations'] = self.relations.pretty()
        if self.relations_history is not None:
            obj['relations_history'] = self.relations_history.pretty()
        return obj

    def get_relations(self, loads=None, table='relation'):
        tmp = Relations().get_left(table=table, left_id=self.request_id, loads=loads)
        if table == 'relation':
            self.relations = tmp
        if table == 'relation_history':
            self.relations_history = tmp


class Requests:

    def __init__(self):
        self.requests = []

    def derive(self):
        return [e.derive() for e in self.requests]

    def pretty(self):
        return [e.pretty() for e in self.requests]

    def get(self, table='request', limit=None, offset=None, loads=None):
        # Database
        if limit is not None:
            r = self.get_limit_sql(table, limit, offset)
        else:
            r = self.get_sql(table, limit)
        # Model
        for el in r:
            request = Request(el[0])  # The first index of fillable is the unique id.
            for idx, e in enumerate(Request.fillables[table]):
                setattr(request, e, el[idx])
            self.requests.append(request)
        # Load
        if loads is not None:
            if 'relation' in loads:
                for el in self.requests:
                    el.get_relations(loads=loads, table='relation')
            if 'relation_history' in loads:
                for el in self.requests:
                    el.get_relations(loads=loads, table='relation_history')

        return self

    def get_sql(self, table='request'):
        query = "SELECT {} FROM `{}`".format(
            ', '.join(["`{}`".format(e) for e in Request.fillables[table]]), table)
        return SQL().all(query, [])

    def get_limit_sql(self, table='request', limit=100, offset=0):
        query = "SELECT {} FROM `{}` LIMIT {} OFFSET {}".format(
            ', '.join(["`{}`".format(e) for e in Request.fillables[table]]), table, limit, offset)
        return SQL().all(query, [])


