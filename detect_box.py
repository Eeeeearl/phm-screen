import numpy as np
import pandas as pd
import psycopg2 as pg2
from utils.box import *
import datetime


cache_dict = {}
result_dict = {}


def main(kwargs):
    """主函数"""
    global cache_dict, result_dict

    # sql
    start_time = kwargs.get('start_time')
    end_time = kwargs.get('end_time')
    if not start_time or not end_time:  # 若不指定起止时间，则用时间间隔
        timedelta = kwargs.get('timedelta', 30)
        start_time = datetime.datetime.now() - datetime.timedelta(days=timedelta)
        end_time = datetime.datetime.now()
    if result_dict:  # 若非首次检测，则只查最近2h
        start_time = datetime.datetime.now() - datetime.timedelta(hours=2)
        end_time = datetime.datetime.now()
    sql_domain = f"select * from public.domain where time >= '{start_time}' and time <= '{end_time}'"
    sql_device = 'select * from public.device'

    # 读表
    conn = kwargs.get('conn')
    domain = read_data(sql_domain, conn)
    device = read_data(sql_device, conn)

    # 建立缓存
    for name in device['name']:
        if name not in cache_dict.keys():
            cache_dict[name] = pd.DataFrame({})

    # 合并
    data = pd.merge(left=domain, right=device[['id', 'name']], how='inner', left_on='device_id', right_on='id')
    data = data.drop('id', axis=1)

    # 清洗
    data = clear_data(data, kwargs)  # unique,long,run,sort,array

    # 重组
    device_dict = restructure(data, kwargs)  # devices dict

    # 缓存
    cache_dict = cash_data(device_dict, cache_dict, kwargs)

    # 检测
    result_dict = detect(cache_dict, result_dict, kwargs)  # results dict

    return result_dict


if __name__ == '__main__':
    # 连接数据库
    conn = pg2.connect(host='127.0.0.1', port=5432, user='postgres', password='postgres', database='postgres')

    main({'conn': conn})

    conn.close()


