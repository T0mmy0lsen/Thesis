from helpers.sql import SQL


class Communication:

    fillables = {
        'communication': ['id', 'message'],
        'communication_history': ['id', 'message']
    }

    def __init__(self, communication_id):
        self.communication_id = communication_id
        self.message = ''

    def pretty(self):
        return {'message': self.message}


class Communications:

    def __init__(self):
        self.communications = []

    def pretty(self):
        return [e.pretty() for e in self.communications]

    def get_sql(self, table='item', communication_id=0):
        query = "SELECT {} FROM `{}` WHERE `id` = %s".format(
            ', '.join(["`{}`".format(e) for e in Communication.fillables[table]]), table)
        return SQL().all(query, [f'{communication_id}'])

    def get(self, table, communication_id):
        # Database
        r = self.get_sql(table=table, communication_id=communication_id)
        # Model
        for el in r:
            item = Communication(el[0])  # The first index of fillable is the unique id.
            for idx, e in enumerate(Communication.fillables[table]):
                setattr(item, e, el[idx])
            self.communications.append(item)
        return self