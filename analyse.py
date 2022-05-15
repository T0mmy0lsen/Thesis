import pandas as pd
import config
from objs.Request import Request


def read_data():
    file_name = 'request_history_communication'
    df = pd.read_csv(f'{config.BASE_PATH}/cache/{file_name}.csv')

    print(df.iloc[1000])
    print(df.iloc[1001])
    print(df.iloc[1002])

def read_object():

    request = Request()