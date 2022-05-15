from helpers.sql import SQL


class Item:

    fillables = {
        'item': ['id', 'description', 'username'],
        'item_history': ['id', 'description', 'username']
    }

    def __init__(self, item_id):
        self.item_id = item_id
        self.description = ''
        self.username = ''


    def pretty(self):
        return {'username': self.username, 'description': self.description}


class Items:

    def __init__(self):
        self.items = []

    def pretty(self):
        return [e.pretty() for e in self.items]

    def get_sql(self, table='item', item_id=0):
        query = "SELECT {} FROM `{}` WHERE `id` = %s".format(
            ', '.join(["`{}`".format(e) for e in Item.fillables[table]]), table)
        return SQL().all(query, [f'{item_id}'])

    def get(self, table, item_id):
        # Database
        r = self.get_sql(table=table, item_id=item_id)
        # Model
        for el in r:
            item = Item(el[0])  # The first index of fillable is the unique id.
            for idx, e in enumerate(Item.fillables[table]):
                setattr(item, e, el[idx])
            self.items.append(item)
        return self
