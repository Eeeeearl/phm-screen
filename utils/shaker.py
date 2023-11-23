import math
from decimal import Decimal
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import traceback


# 清理数据
def clear_date(dataframe, name="", flag=False):
    if dataframe.isnull().sum().sum() != 0:
        dataframe.dropna(how='any', axis=0, inplace=True)  # 删除存在空值数据
    if any(dataframe.duplicated()):
        dataframe.drop_duplicates(inplace=True)  # 删除重复项
    if flag:
        dataframe['name'] = name
    return dataframe


# 整理数据
def neaten_data(dataframe):
    df = dataframe.copy()
    df['maoci'] = False
    df['gongzhen'] = False
    df['pianzhen'] = False
    df['niuzhen'] = False
    df['mosun'] = False
    df['diantou'] = False
    df = df.sort_values(by='time')  # 时间排序
    df = df.reset_index(drop=True)  # 重置下标
    return df


# 按pos拆分数据
def split_df(dataframe):
    pos_uni = dataframe['pos'].unique()  # unique pos: [c1,c2]
    df_pos = {}
    for pos in pos_uni:
        df_pos[pos] = neaten_data(dataframe[dataframe['pos'].isin([pos])])
    return df_pos, pos_uni


# 加速度最值 gongzhen
def judge_acc(data, kwargs: dict):
    ax_range = kwargs['ax_range']
    ay_range = kwargs['ay_range']
    az_range = kwargs['az_range']
    mul = kwargs['mul']
    if az_range is None:
        az_range = [-34, 48]
    if ay_range is None:
        ay_range = [-6, 6]
    if ax_range is None:
        ax_range = [-15, 3]
    if not isinstance(ax_range, list) or not isinstance(ay_range, list) or not isinstance(az_range, list) or \
            len(ax_range) != 2 or len(ay_range) != 2 or len(az_range) != 2:
        return data

    ax_mid = (ax_range[0]+ax_range[1])/2
    ay_mid = (ay_range[0]+ay_range[1])/2
    az_mid = (az_range[0]+az_range[1])/2
    ax_dis = (ax_range[1] - ax_mid) * mul
    ay_dis = (ay_range[1] - ay_mid) * mul
    az_dis = (az_range[1] - az_mid) * mul

    for i in data:  # c1 c2
        dataframe = data[i]
        dataframe.loc[(dataframe['ax'] <= (ax_mid - ax_dis)) | (dataframe['ax'] >= (ax_mid + ax_dis)),
                      'gongzhen'] = True
        dataframe.loc[(dataframe['ay'] <= (ay_mid - ay_dis)) | (dataframe['ay'] >= (ay_mid + ay_dis)),
                      'gongzhen'] = True
        dataframe.loc[(dataframe['az'] <= (az_mid - az_dis)) | (dataframe['az'] >= (az_mid + az_dis)),
                      'gongzhen'] = True
    return data


# 欧拉角最值判断 diantou pianzhen niuzhen
def judge_euler(data, kwargs: dict):
    pitch_range = kwargs['pitch_range']
    roll_range = kwargs['roll_range']
    yaw_range = kwargs['yaw_range']
    if pitch_range is None:
        pitch_range = [-90, 90]
    if roll_range is None:
        roll_range = [-180, 180]
    if yaw_range is None:
        yaw_range = [-180, 180]
    if not isinstance(pitch_range, list) or not isinstance(roll_range, list) or not isinstance(yaw_range, list) or \
            len(pitch_range) != 2 or len(roll_range) != 2 or len(yaw_range) != 2:
        return data

    for i in data:  # c1 c2
        dataframe = data[i]
        dataframe.loc[(dataframe['pitch'] <= pitch_range[0]) | (dataframe['pitch'] >= pitch_range[1]), 'diantou'] = True
        dataframe.loc[(dataframe['roll'] <= roll_range[0]) | (dataframe['roll'] >= roll_range[1]), 'pianzhen'] = True
        dataframe.loc[(dataframe['yaw'] <= yaw_range[0]) | (dataframe['yaw'] >= yaw_range[1]), 'niuzhen'] = True

    return data


# 温度最值判断 mosun
def judge_temp(data, kwargs):
    temp_range = kwargs['temp_range']
    if temp_range is None:
        temp_range = [0, 55]
    if not isinstance(temp_range, list) or len(temp_range) != 2:
        return data
    for i in data:  # c1 c2
        dataframe = data[i]
        dataframe.loc[dataframe['temperature'] > temp_range[1], 'mosun'] = True
    return data


# 判断单位时间内的温差变化  mosun
def judge_diff_temp(data, kwargs):
    temp = kwargs.get('temp', None)  # 对比的单位时间内的温度差，单位：°
    time = kwargs.get('time', 30)  # 对比的数据间时间差，单位：min
    nor_interval = kwargs.get('nor_interval', 60)  # 单位：second
    if temp is None:
        temp = time * 0.5  # 温差最大限额处于 -0.5~0.5/60s 之间

    for q in data:  # c1 c2
        df = data[q]
        # 增加时间差列、温差列
        len_time = len(df['time'])
        len_temp = len(df['temperature'])
        if len_time == 0 or len_temp == 0:
            continue
        # 时间差
        diff_time = [nor_interval]  # 时间差第一位默认指定为规范间隔时间：60s
        for i in range(1, len_time, 1):
            time2 = df['time'][i]
            time1 = df['time'][i - 1]
            diff_time.append(time2.timestamp() - time1.timestamp())
        # 温度差
        diff_temp = [0]  # 温差第一位默认为：0
        for i in range(1, len_temp, 1):
            temp2 = df['temperature'][i]
            temp1 = df['temperature'][i - 1]
            diff_temp.append(temp2 - temp1)
        df['diff-time'] = diff_time
        df['diff-temp'] = diff_temp
        # 间隔大于设定时间的，进行分段判断  # 返回的是下标
        indexes = np.argwhere(np.array(diff_time) > time * 60).flatten().tolist()
        # 是否存在温差过快的情况，true 不存在，false 存在
        if indexes == [] or indexes is None:
            # 整个数据分析
            time_arr = df['time'].to_list()
            temp_arr = df['temperature'].to_list()
            # 对应关系：第一个代表第一个时间增加间隔时间内的温差信息，可通过下标位置，定位到时间，定位到整个时间间隔内的行数据
            arr1 = com_diff_temp(time_arr, temp_arr, time)  # [.., .., .., 0, .., 0,...]
            # 得到下标，该下标超过稳定温度了
            index_list1 = np.argwhere((np.array(arr1) >= temp)).flatten().tolist()
            if index_list1:
                # 温度上升过快
                for ind in index_list1:  # add time and value
                    df.iloc[ind: ind+30, df.columns.get_loc('mosun')] = True
        else:
            # 分时间段进行分析
            if len(indexes) == 1:
                arr1 = com_diff_temp(df['time'].to_list()[: indexes[0]],
                                     df['temperature'].to_list()[: indexes[0]], time)
                index_list1 = np.argwhere((np.array(arr1) >= temp)).flatten().tolist()
                if index_list1:
                    # 温度上升过快
                    for ind in index_list1:  # add time and value
                        df.iloc[ind: ind + 30, df.columns.get_loc('mosun')] = True
                arr2 = com_diff_temp(df['time'].to_list()[indexes[0]:], df['temperature'].to_list()[indexes[0]:], time)
                index_list3 = np.argwhere(np.array(arr2) >= temp).flatten().tolist()
                if index_list3:
                    # 温度上升过快
                    for ind in index_list3:  # add time and value
                        df.iloc[ind+indexes[0]: ind+30+indexes[0], df.columns.get_loc('mosun')] = True
            else:
                for i in range(len(indexes)):
                    if i == 0:
                        time_arr = df['time'].to_list()[: indexes[i]]
                        temp_arr = df['temperature'].to_list()[: indexes[i]]
                        add_index = 0
                    elif i == len(indexes) - 1:
                        time_arr = df['time'].to_list()[indexes[i]:]
                        temp_arr = df['temperature'].to_list()[indexes[i]:]
                        add_index = indexes[i]
                    else:
                        time_arr = df['time'].to_list()[indexes[i-1]: indexes[i]]
                        temp_arr = df['temperature'].to_list()[indexes[i-1]: indexes[i]]
                        add_index = indexes[i-1]

                    arr1 = com_diff_temp(time_arr, temp_arr, time)
                    index_list1 = np.argwhere(np.array(arr1) >= temp).flatten().tolist()
                    if index_list1:
                        # 温度上升过快
                        for ind in index_list1:  # add time and value
                            df.iloc[ind+add_index: ind+30+add_index, df.columns.get_loc('mosun')] = True
        # ADD: 对单条数据进行判断
        df[(df['diff-time'] >= 55) & (df['diff-time'] <= 65) &
           (df['diff-temp'] >= 1)].iloc[:, df.columns.get_loc('mosun')] = True
    return data


# 返回每单位时间内的温差
def com_diff_temp(time_arr, temp_arr, time=30):
    """
    :param time_arr:    时间列表
    :param temp_arr:    温度列表
    :param time:        时间差：默认30min
    :return:            单位时间内的温差列表
    """
    result = []
    last_time = time_arr[-1]
    for i in range(len(time_arr)):
        target_time = time_arr[i] + relativedelta(minutes=time)
        if target_time <= last_time:
            j = find_time_index(target_time, time_arr[i:])
            if j is None or j == -1:
                result.append(0)
            else:
                result.append(temp_arr[i+j] - temp_arr[i])
        else:
            break
    return result


# 查找与目标时间接近的时间下标
def find_time_index(target_time, arr):
    if len(arr) < 2:
        return None
    for i in range(len(arr)):
        if target_time < arr[i]:
            if arr[i] - target_time >= target_time - arr[i-1]:
                return i-1
            else:
                return i
    return -1


def predict_yis(dataframe, olj_model, temp_model) -> pd.DataFrame:
    dataframe = dataframe.sort_values(by='time')  # 时间排序
    dataframe = dataframe.reset_index(drop=True)  # 重置下标
    df = dataframe[-30:]
    # df = df.sort_values(by='time')  # 时间排序
    # df = df.reset_index(drop=True)  # 重置下标
    if len(df[df['diff-time'] >= 70]) == 0:  # 皆符合要求
        olj_input = df[['pitch', 'roll', 'yaw']].to_numpy()
        olj_output = olj_model.predict(olj_input)
        # output_data = [[0, 0, 0]]  # test data
        q0,q1,q2,q3 = [],[],[],[]
        pit,rol,yaw = [],[],[]
        for i in range(olj_output.shape[0]):
            # 欧拉角
            pit.append(Decimal(olj_output[i][0].item()).quantize(Decimal('0.0000000')))
            rol.append(Decimal(olj_output[i][1].item()).quantize(Decimal('0.0000000')))
            yaw.append(Decimal(olj_output[i][2].item()).quantize(Decimal('0.0000000')))
            # 四元数
            q_data = olj2sys(olj_output[i])
            q0.append(Decimal.from_float(q_data[0]).quantize(Decimal('0.0000000')))
            q1.append(Decimal.from_float(q_data[1]).quantize(Decimal('0.0000000')))
            q2.append(Decimal.from_float(q_data[2]).quantize(Decimal('0.0000000')))
            q3.append(Decimal.from_float(q_data[3]).quantize(Decimal('0.0000000')))

        temp_input = df[['pitch','roll','yaw','q0','q1','q2','q3','temperature']].to_numpy()
        temp_output = temp_model.predict(temp_input)
        temp = []
        for i in range(temp_output.shape[0]):
            temp.append(Decimal(temp_output[i][-1].item()).quantize(Decimal('0.00')))
        return pd.DataFrame({'time': [df['time'].iloc[-1]+timedelta(minutes=1)],
                             'code': [df['code'].iloc[-1]],
                             'ax': [[0]], 'ay': [[0]], 'az': [[0]], 'wx': [[0]], 'wy': [[0]], 'wz': [[0]],
                             'pitch': [pit], 'roll': [rol], 'yaw': [yaw],
                             'q0': [q0], 'q1': [q1], 'q2': [q2], 'q3': [q3],
                             'temperature': [temp],
                             'device_id': [df['device_id'].iloc[-1]],
                             'pos': [df['pos'].iloc[-1]]})
    return pd.DataFrame({'time':[], 'code':[], 'ax':[], 'ay':[], 'az':[], 'wx':[], 'wy':[], 'wz':[],
                         'pitch':[], 'roll':[], 'yaw':[], 'q0':[], 'q1':[], 'q2':[], 'q3':[], 'temperature':[],
                         'device_id':[], 'pos':[]})


# 通过时间字段获取数据库中的yis数据
def get_yis(kwargs, device, result_shaker, device_name):
    """
    @param kwargs:          config info
    @param device:          device id
    @param result_shaker:   global result_shaker of the main function
    @param device_name:     device name
    @return:                the data of yis
    """
    if kwargs['conn'] is None or device is None:
        return None
    table_name = "public.{}".format('yis')
    daydelta = kwargs.get('daydelta', 1)
    start_time = kwargs.get('start_time', '')
    end_time = kwargs.get('end_time', '')

    over_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') if end_time == '' else end_time
    if start_time == '':
        if not result_shaker:  # 如果result_shaker为空，采用daydelte的值
            begin_time = (datetime.now() - timedelta(days=daydelta)).strftime('%Y-%m-%d %H:%M:%S.%f')
        else:  # 如果不为空，则采用间隔31分钟
            begin_time = (datetime.now() - timedelta(minutes=60)).strftime('%Y-%m-%d %H:%M:%S.%f')
    else:
        begin_time = (datetime.now() - timedelta(minutes=60)).strftime('%Y-%m-%d %H:%M:%S.%f')
    sql_command = f"select * from {table_name} where time>'{begin_time}' and time<'{over_time}' "\
                  f"and device_id='{device}'"
    # print(f'get_yis函数运行的sql语句为：{sql_command}.')

    try:
        data = pd.read_sql(sql_command, kwargs['conn'])
        if data.empty:
            # print(f"{device}>>> No suitable shaker data found.")
            return None
        else:
            if not result_shaker.get(str(device_name), pd.DataFrame({})).empty:
                if 'time' in result_shaker.get(str(device_name)):
                    if len(result_shaker[str(device_name)]['time']) > 0:
                        if len(data[data.time > result_shaker[str(device_name)]['time'].iloc[-1]]) == 0:
                            return None
        return data
    except Exception as e:
        print(f"SQL run fail, exception info: {e}.")
        print(f'Info:{traceback.format_exc()}')
        return None


def olj2sys(olj: list):
    p, r, y = olj[:]
    sinp = math.sin(math.radians(p / 2))
    siny = math.sin(math.radians(y / 2))
    sinr = math.sin(math.radians(r / 2))

    cosp = math.cos(math.radians(p / 2))
    cosy = math.cos(math.radians(y / 2))
    cosr = math.cos(math.radians(r / 2))

    w = cosr * cosp * cosy + sinr * sinp * siny
    x = sinr * cosp * cosy - cosr * sinp * siny
    y = cosr * sinp * cosy + sinr * cosp * siny
    z = cosr * cosp * siny - sinr * sinp * cosy
    return [w,x,y,z]


def detect(df_dict, kwargs, pos_list):
    # 判断加速度最值
    df_dict = judge_acc(data=df_dict, kwargs=kwargs)
    # 判断欧拉角最值
    df_dict = judge_euler(data=df_dict, kwargs=kwargs)
    # 判断温度最值
    df_dict = judge_temp(data=df_dict, kwargs=kwargs)
    # 判断单位时间内温差变化
    df_dict = judge_diff_temp(data=df_dict, kwargs=kwargs)

    if len(pos_list) == 2:
        result = pd.concat([df_dict[pos_list[0]], df_dict[pos_list[1]]])
        result = result.sort_values(by='time')
        result = result.reset_index(drop=True)
    else:
        result = df_dict[pos_list[0]]
    # 规范输出数据
    # result = result[['time', 'name', 'pos', 'maoci', 'gongzhen', 'pianzhen', 'niuzhen', 'mosun', 'diantou']]
    return result


# ADD: predict function
def predict(dataframe, poses: list, model1, model2):
    # 数据判断
    # 1、判断量满足30？
    # 2、判断最后30个数据满足时间差均小于70s
    # 3、后期增加时间差不满足皆小于70s的处理方式（可以做 等差 减少空白数据）
    result = pd.DataFrame({'time':[], 'code':[], 'ax':[], 'ay':[], 'az':[], 'wx':[], 'wy':[], 'wz':[],
                           'pitch':[], 'roll':[], 'yaw':[], 'q0':[], 'q1':[], 'q2':[], 'q3':[], 'temperature':[],
                           'device_id':[], 'pos':[]})
    for pos in poses:
        df = dataframe[dataframe.pos==pos]
        if len(dataframe[dataframe.pos==pos]) >= 30:
            pre_dict = predict_yis(df, model1, model2)  # return dict
            result = pd.concat([result, pre_dict])
    return result
