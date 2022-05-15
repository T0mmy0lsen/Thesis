import pymysql as pymysql


class SQL:

    # Return all results
    def all(self, sql, values=None):
        if values is None:
            values = []
        self._cursor.execute(sql, values)
        result = self._cursor.fetchall()
        return result

    def count(self, sql):
        self._cursor.execute(sql)
        return self._cursor.fetchone()[0]

    # Return first row in result
    def one(self, sql, values=None):
        if values is None:
            values = []
        self._cursor.execute(sql, values)
        result = self._cursor.fetchone()
        return result

    # Simple insert statement
    def insert(self, table, el):
        sql = "INSERT INTO `{}` VALUES ({})".format(table, ', '.join(['%s'] * len(el)))
        self._cursor.execute(sql, el)

    # Commit query
    def commit(self):
        self._connection.commit()

    # Close connection
    def close(self):
        self._connection.close()

    def __init__(self, db='ihlp'):
        self._connection = pymysql.connect(host='localhost', user='root', password='', db=db)
        self._cursor = self._connection.cursor()
