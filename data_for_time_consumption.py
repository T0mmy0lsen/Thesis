import numpy as np
import pandas as pd

import config
from logic.Prepare_v2 import Prepare_v2
from datetime import datetime


def secondsPast(time, hour):
    point = time.replace(hour=hour, minute=0, second=0, microsecond=0)
    return (time - point).total_seconds()


def secondsUntil(time, hour):
    point = time.replace(hour=hour, minute=0, second=0, microsecond=0)
    return (point - time).total_seconds()


def solutionTime(start, end):
    if not isinstance(start, str) or not isinstance(end, str):
        return -1

    if start[0] == '0' or end[0] == '0':
        return -1

    dt_start = datetime.fromisoformat(start)
    dt_end = datetime.fromisoformat(end)

    total = dt_end - dt_start

    np_days_full = np.busday_count(start[0:10], end[0:10], weekmask=[1, 1, 1, 1, 1, 1, 1], holidays=[])
    np_days_buss = np.busday_count(start[0:10], end[0:10], weekmask=[1, 1, 1, 1, 1, 0, 0], holidays=['2017-01-01'])
    np_days_weekend = np_days_full - np_days_buss

    subtract_seconds = (np_days_weekend * 24 * 60 * 60) + (np_days_buss * 16 * 60 * 60)

    end_to_0800 = 0
    end_after_1600 = 0
    start_to_0800 = 0
    start_after_1600 = 0

    seconds_until = secondsUntil(dt_start, 8)
    if seconds_until > 0:
        start_to_0800 = seconds_until

    seconds_until = secondsPast(dt_start, 16)
    if seconds_until > 0:
        start_after_1600 = seconds_until

    seconds_until = secondsUntil(dt_end, 8)
    if seconds_until > 0:
        end_to_0800 = seconds_until

    seconds_until = secondsPast(dt_end, 16)
    if seconds_until > 0:
        end_after_1600 = seconds_until

    solution = total.total_seconds() + end_to_0800 - end_after_1600 - start_to_0800 - start_after_1600 - subtract_seconds

    solution = int(solution)
    if solution > 0:
        return int(solution / 60)

    return 0


def features(r, rh):
    """
    The goal is to extract meaning full data other than just text.
    The features should enable the ability to train a model that can predict:

        Response    time until the Request is being processed the first time.
        Idle        time waited due to customer response.
        Solution    time spend solving the issue.

    Things to consider:

        Working hours   We could consider time to be 8 - 16, monday to friday.
                        This means a request received at 15:55, and solved the next day at 08:05 would have 10 min solve time.
        Workload        If there are many Request to be solved it is logical that customers should wait longer for the Request to be solved.
    """

    tmp = []
    for x in r:

        time_at_solved = x.solutionDate
        time_at_received = x.receivedDate
        time_at_reaction = 0
        time_at_communication_last = 0
        time_at_communication_first = 0

        data = rh[rh['leftId'] == x.id]

        for jdx, relation in data.iterrows():
            if relation['rightId'] == 1935820:
                time_at_reaction = relation['tblTimeStamp']
            if relation['rightType'] == 'CommunicationSimple':
                time_at_communication_last = relation['tblTimeStamp']
                if time_at_communication_first == 0:
                    time_at_communication_first = time_at_communication_last

        if isinstance(time_at_solved, datetime):
            time_at_solved = time_at_solved.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(time_at_received, datetime):
            time_at_received = time_at_received.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(time_at_reaction, datetime):
            time_at_reaction = time_at_reaction.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(time_at_communication_last, datetime):
            time_at_communication_last = time_at_communication_last.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(time_at_communication_first, datetime):
            time_at_communication_first = time_at_communication_first.strftime("%Y-%m-%d %H:%M:%S")

        minutes_until_reaction = solutionTime(time_at_received, time_at_reaction)
        minutes_until_communication_last = 0
        minutes_until_communication_first = 0

        if time_at_communication_last != 0:
            minutes_until_communication_first = solutionTime(time_at_received, time_at_communication_first)
            minutes_until_communication_last = solutionTime(time_at_received, time_at_communication_last)
            minutes_until_solved = minutes_until_communication_last
        else:
            minutes_until_solved = solutionTime(time_at_received, time_at_solved)

        request_id = x.id
        tmp.append({
            'id': f'{request_id}',
            'solvedTime': f'{minutes_until_solved}',
            'reactionTime': f'{minutes_until_reaction}',
            'communicationLast': f'{minutes_until_communication_last}',
            'communicationFirst': f'{minutes_until_communication_first}'
        })

    tmp = pd.DataFrame(tmp, columns=['id', 'solvedTime', 'reactionTime', 'communicationLast', 'communicationFirst'])
    return tmp


def build():
    df = pd.DataFrame([], columns=['id', 'solvedTime', 'reactionTime', 'communicationLast', 'communicationFirst'])
    r = pd.read_csv('data/request.csv')
    rh = pd.read_csv('data/request_relation_history.csv')

    limit = 1000
    limit_max = int(86100 / limit)

    for i in range(0, limit_max + 1):
        d = features(r.requests, rh)
        df = pd.concat([df, d])
        print(i * 1000)

    df.to_excel('output/time_consumption.xlsx', index=False)


def timeConsumptionInspection():
    timeConsumption = pd.read_excel(f'{config.BASE_PATH}/time_consumption.xlsx')

    table = 'request_tasktype_simple'
    p = Prepare_v2(table=table)
    df = p.fetch()
    df = pd.merge(df, timeConsumption, how='inner')
    df = df[df['timeConsumption'] >= 0]

    df_50 = df[df['timeConsumption'] < 50000]
    df_25 = df[df['timeConsumption'] < 25000]

    import matplotlib.pyplot as plt

    xpoints = df['id'].to_numpy()
    ypoints = df['timeConsumption'].to_numpy()
    plt.plot(xpoints, ypoints)
    plt.show()

    xpoints = df_25['id'].to_numpy()
    ypoints = df_25['timeConsumption'].to_numpy()
    plt.plot(xpoints, ypoints)
    plt.show()

    xpoints = df_50['id'].to_numpy()
    ypoints = df_50['timeConsumption'].to_numpy()
    plt.plot(xpoints, ypoints)
    plt.show()


def main():
    # Build timeConsumption data
    # assert testSolutionTime()
    # build()
    pass


def testSolutionTime():
    return (
            solutionTime('2022-05-13 15:59', '2022-05-16 08:00') == 1 and
            solutionTime('2022-05-13 15:59', '2022-05-16 16:05') == 481 and
            solutionTime('2022-05-13 15:59', '2022-05-16 15:00') == 421 and
            solutionTime('2022-05-13 07:00', '2022-05-16 15:00') == 900
    )


if __name__ == '__main__':
    main()
