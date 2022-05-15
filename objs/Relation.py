from helpers.sql import SQL
from objs.Communication import Communications
from objs.Item import Item, Items
from objs.Object import Object


class Relation:

    fillables = {
        'relation': ['id', 'leftId', 'rightId', 'leftType', 'rightType'],
        'relation_history': ['id', 'leftId', 'rightId', 'leftType', 'rightType', 'tblTimeStamp']
    }

    def __init__(self, relation_id):
        self.relation_id = relation_id
        self.items = None
        self.communications = None
        self.id = 0
        self.rightId = 0
        self.rightType = ''
        self.tblTimeStamp = None

    def pretty(self):
        obj = {'id': self.id, 'rightId': self.rightId}
        if self.tblTimeStamp is not None:
            obj['timestamp'] = self.tblTimeStamp.strftime('%Y-%m-%d %H:%M')
        if self.rightType is not None:
            obj['type'] = self.rightType
        if self.items is not None:
            obj['right'] = self.items.pretty()
        if self.communications is not None:
            obj['right'] = self.communications.pretty()
        return obj

    def get_right(self):
        if self.rightType[0:4] == 'Item':
            self.items = Items().get('item', self.rightId)
        if self.rightType[0:13] == 'Communication':
            self.communications = Communications().get('communication', self.rightId)


class Relations:

    def __init__(self):
        self.relations = []

    def pretty(self):
        lst = [e.pretty() for e in self.relations]
        lst.sort(key=lambda x: x['id'], reverse=False)
        tmp = []
        keys = []
        for x in lst:
            if x['rightId'] not in keys:
                tmp.append(x)
                keys.append(x['rightId'])
        return tmp

    def get_left(self, table='relation', left_id=0, loads=None):
        # Database
        r = self.get_left_sql(table, left_id)
        # Model
        for el in r:
            relation = Relation(el[0])  # The first index of fillable is the unique id.
            for idx, e in enumerate(Relation.fillables[table]):
                setattr(relation, e, el[idx])
            self.relations.append(relation)
        # Load
        if loads is not None:
            if 'right' in loads:
                for el in self.relations:
                    el.get_right()
        return self

    def get_left_sql(self, table='relation', left_id=0):
        query = "SELECT {} FROM `{}` WHERE `leftId` = %s".format(
            ', '.join(["`{}`".format(e) for e in Relation.fillables[table]]), table)
        return SQL().all(query, [f'{left_id}'])


