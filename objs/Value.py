from helpers.sql import SQL
from objs.Object import Object


class Value:

    _sql = None
    _amount = None

    def __init__(self, amount):
        self._sql = SQL('data')
        self._amount = amount

    def all(self, key):
        if self._amount is None:
            query = "SELECT `id`, `key`, `value` FROM `values` WHERE `key` = %s"
        else:
            query = f"SELECT `id`, `key`, `value` FROM `values` WHERE `key` = %s LIMIT {self._amount}"
        result = self._sql.all(query, [key])
        return result

    def set(self, el):
        self._sql.insert('values', el)
        self._sql.commit()

    def sql(self):
        return self._sql
