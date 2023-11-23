from datetime import datetime, timedelta
import json
import time
import pandas as pd
from utils.shaker import *
from utils.common import get_device_id_name


result_shaker = {}  # 结果数据


def main(kwargs: dict, model_olj, model_temp):
    global result_shaker

    id_name = get_device_id_name(kwargs['conn'])
    # print(id_name)
    results = {}  # 多device结果
    pre_results = pd.DataFrame({})  # 预测结果
    for _id in id_name:  # 对各个id设备进行检测、预测
        name = id_name[_id]
        # 获取数据
        df = get_yis(kwargs, _id, result_shaker, name)
        if df is None:
            result = pd.DataFrame({})  # 单device结果
            # return None
        else:
            # 数据清理
            df = clear_date(df, name, True)
            # 按pos拆分 增加异常列
            df_dict, poses = split_df(df)
            # 检测  整理好的数据
            result = detect(df_dict, kwargs, poses)
            # print(result)
            # ADD: 预测
            pre_result = predict(result, poses, model_olj, model_temp)
            pre_results = pd.concat([pre_result, pre_results])
            # print('预测数据：', pre_results)
            # print('预测数据。')
        # 结果合并
        if len(result_shaker.get(str(name), pd.DataFrame({}))) > 0:
            df_cd = clear_date(pd.concat([result_shaker[str(name)], result]))
            results[str(name)] = df_cd[0 if len(df_cd)-1440<=0 else len(df_cd)-1440: ]  # 只保留最近的一天数据，默认一分钟一条
        else:
            results[str(name)] = result

    result_shaker = results

    # print(result_shaker)
    return result_shaker, pre_results


if __name__ == '__main__':
    from utils.common import connect_db2
    from utils import model
    # pg 连接器
    db_kwargs = {
        'database': 'postgres',
        'user': 'postgres',
        'password': '123456',
        'host': '127.0.0.1',
        'port': 5432,
    }
    engine, conn = connect_db2(db_kwargs)

    model_olj = model.BiLSTM('./model/BiLSTM_pry_pre5.h5', pattern='olj')
    model_temp = model.BiLSTM('./model/BiLSTM_temp_pre5.h5', pattern='temp')

    previous_time = datetime.strptime('2023-10-18 00:00:00.0', '%Y-%m-%d %H:%M:%S.%f')  # 起始时间
    current_time = previous_time + timedelta(minutes=31)  # 31分钟后的时间
    # start_time = previous_time.strftime('%Y-%m-%d %H:%M:%S.%f')
    # end_time = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')
    print(previous_time, current_time)

    for i in range(10):
        # current_time = datetime.now()
        previous_time = previous_time + timedelta(minutes=1)
        current_time = current_time + timedelta(minutes=1)
        kwargs = {
            # 'start_time': '2023-04-22 00:00:00.0',
            # 'stop_time': '2023-04-26 00:00:00.0',
            'start_time': previous_time,
            'end_time': current_time,
            "daydelta": 1,
            "az_range": [-34, 48],
            "ay_range": [-6, 6],
            "ax_range": [-15, 3],

            "pitch_range": [-90, 90],
            "roll_range": [-180, 180],
            "yaw_range": [-180, 180],
            "temp_range": [0, 60],

        }
        shaker_kwargs = {
            'conn': conn,
            'start_time': kwargs.get('start_time'),
            'end_time': kwargs.get('end_time'),
            'daydelta': kwargs.get('daydelta', 1),
            'ax_range': kwargs.get('ax_range'),
            'ay_range': kwargs.get('ay_range'),
            'az_range': kwargs.get('az_range'),
            'mul': kwargs.get('mul', 1.5),

            'pitch_range': kwargs.get('pitch_range'),
            'roll_range': kwargs.get('roll_range'),
            'yaw_range': kwargs.get('yaw_range'),

            'temp_range': kwargs.get('temp_range'),

            'temp': kwargs.get('temp', None),
            'time': kwargs.get('time', 30),
            'nor_interval': kwargs.get('nor_interval', 60),
        }
        time.sleep(60)  # 模拟等待 1 min

        print('shaker_kwargs', shaker_kwargs)
        main(shaker_kwargs, model_olj, model_temp)
        # break
    # conn.close()
