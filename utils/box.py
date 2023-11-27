import numpy as np
import pandas as pd


# ====================================================
#                      数据处理
# ====================================================
def read_data(sql, conn):
    """读取数据"""
    return pd.read_sql(sql, conn)


def format_data(data):
    """格式化数据"""
    # time: Timestamp
    # code: str
    # x_time: list(Decimal), need to transform list to array
    # x_frequency: list(Decimal), need to transform list to array
    # device_id: str
    # pos: str
    if len(data) == 0:
        return data
    columns = []
    for col in data.columns:
        if isinstance(data[col].iloc[0], list):
            columns.append(col)
    # list to array
    data[columns] = data.loc[:, columns].applymap(lambda x: np.array(x))
    return data


def clear_data(data, kwargs={}):
    """清洗数据"""
    len_thr = kwargs.get('len_thr', 1000)
    std_thr = kwargs.get('std_thr', 825)
    # unique
    data = data.drop_duplicates(['time', 'code', 'device_id', 'pos'], keep='first')
    # long (>=1000)
    long = data[data.loc[:, 'x_time'].map(lambda x: len(x)) >= len_thr]
    # run (std>825)
    std = long['x_time'].apply(lambda x: np.std(x))
    run = long[long.index.isin(std[std > std_thr].index)]  # not detect the stop of status
    # stop = long[long.index.isin(std[std <= std_thr].index)]  # detect the stop of status
    # sort
    run = run.sort_values('time')
    # stop = stop.sort_values('time')
    # array
    run = format_data(run)
    # stop = format_data(stop)
    return run


def cash_data(device_dict, cache_dict, kwargs):
    """按设备缓存"""
    timedelta = kwargs.get('timedelta', 30)
    length = kwargs.get('room', 4*24*timedelta)
    for name in device_dict.keys():
        cache = cache_dict[name]
        device = device_dict[name]
        if len(cache) == 0:
            cache = device.iloc[-length:, :]
        elif list(cache.columns.values) != list(device.columns.values):
            print(f'>>> {device["name"].iloc[0]}: 表头不一致，缓存失败！')
        else:
            last_time = cache.iloc[-1, :]['time']
            data = device[device['time'] > last_time]
            cache = pd.concat([cache, data])
            cache = cache.iloc[-length:, :]
        cache_dict[name] = cache
    return cache_dict


def temp(df, time, gap, temp):
    """合并一行"""
    timedelta = abs(((df['time'] - time) / (1e9 * 60)).values.astype(np.int32))  # -> minutes
    idx = np.argmin(timedelta)
    if timedelta[idx] < gap:
        temp = pd.concat([temp, df[df['time'] == df['time'].values[idx]]])
    else:
        temp = pd.concat([temp, pd.DataFrame([[0] * len(df.columns)], columns=df.columns)])
    return temp


def restructure(data, kwargs={}):
    """重组数据"""
    # devices
    names = data['name'].unique()
    devices = [data[data['name'] == name] for name in names]

    # concat
    devs = []
    for i, device in enumerate(devices):
        # split by pos
        l1 = device[device['pos'] == 'l1']
        l2 = device[device['pos'] == 'l2']
        r1 = device[device['pos'] == 'r1']
        r2 = device[device['pos'] == 'r2']
        if len(l1) == 0 or len(l2) == 0 or len(r1) == 0 or len(r2) == 0:
            continue

        # concat by time
        gap = kwargs.get('time_gap', 50)  # 正常间隔再56-62min
        temp_l1 = l1.copy()
        temp_l2 = pd.DataFrame([], columns=l1.columns)
        temp_r1 = pd.DataFrame([], columns=l1.columns)
        temp_r2 = pd.DataFrame([], columns=l1.columns)

        for j, time in enumerate(l1['time']):
            temp_l2 = temp(l2, time, gap=gap, temp=temp_l2)
            temp_r1 = temp(r1, time, gap=gap, temp=temp_r1)
            temp_r2 = temp(r2, time, gap=gap, temp=temp_r2)

        temp_l2['time'] = pd.to_datetime(temp_l2['time'])
        temp_r1['time'] = pd.to_datetime(temp_r1['time'])
        temp_r2['time'] = pd.to_datetime(temp_r2['time'])

        temp_l1 = temp_l1.reset_index(drop=True)
        temp_l2 = temp_l2.reset_index(drop=True)
        temp_r1 = temp_r1.reset_index(drop=True)
        temp_r2 = temp_r2.reset_index(drop=True)

        dev = pd.concat([temp_l1, temp_l2, temp_r1, temp_r2], axis=1)
        l1_columns = [col + '_l1' for col in l1.columns]
        l2_columns = [col + '_l2' for col in l1.columns]
        r1_columns = [col + '_r1' for col in l1.columns]
        r2_columns = [col + '_r2' for col in l1.columns]
        columns = l1_columns + l2_columns + r1_columns + r2_columns
        dev.columns = columns
        dev = dev[(dev['time_l2'] > pd.to_datetime(0)) &
                  (dev['time_r1'] > pd.to_datetime(0)) &
                  (dev['time_r2'] > pd.to_datetime(0))]
        devs.append(dev)
    return dict(zip(names, devs))


# ====================================================
#                      异常检测
# ====================================================
def standardize(arr):
    """把所有数据归一到均值为0方差为1的数据中"""
    arr = np.array(arr)
    mean = arr.mean()
    std = arr.std()
    return (arr - mean) / std


def difference(arr):
    """计算均值与中值的偏离度"""
    arr = np.array(arr)
    mean = arr.mean()
    middle = (arr.max() + arr.min()) / 2
    return abs(middle - mean), mean, middle


def concat_data(data, col=None):
    """拼接数据"""
    concat = np.zeros(0)
    if isinstance(data, pd.DataFrame):
        for row in data.index:
            concat = np.append(concat, data[col][row])
    elif isinstance(data, pd.Series):
        for row in data.index:
            concat = np.append(concat, data[row])
    elif isinstance(data, np.ndarray):
        for arr in data:
            concat = np.append(concat, arr)
    elif isinstance(data, list):
        for lst in data:
            concat = np.append(concat, np.array(lst))
    return concat


# 对一条数据进行时域数据异常检测
def one_time(data: np.ndarray, kwargs, model_IF, result):
    """检测四个振动筛xyz三轴时域数据"""
    outlier = kwargs.get('outRatio', 0.2)
    minData = kwargs.get('minData', 800)
    num = outlier * minData
    frame = pd.DataFrame({})
    for i in range(len(data)):
        frame[i] = data[i]
    pre_frame = model_IF.predict(frame)  # i_anomaly: [1,1,-1,-1,...]
    isNoOut = True
    for i in range(len(data)):
        field = '{}_anomaly'.format(str(i))
        if num <= pre_frame[field].apply(lambda x: sum(x == -1)).sum():
            isNoOut = False
    result['isAbnormal'] = 0 if isNoOut else 1  # 0 False 无异常, 1 True
    return result


def one_cell(data: np.ndarray, kwargs={}):
    """检测一个数据"""
    step = kwargs.get('step', 1000 // 15)  # 步长：1个周期长度（1000：数据量，15：周期数）
    lens = kwargs.get('lens', 1024)  # 传感器采样频率（1s采样数据量）
    diff = kwargs.get('diff', 1/2)  # 若传值<1则计算振幅比例值，若传值>1则使用该值
    # 数据平滑处理 -> 方便得到真实振幅
    smooth_data = np.convolve(data, np.ones((10,))/10, mode='valid')
    minimum_2 = np.min(smooth_data)
    maximum_2 = np.max(smooth_data)
    avgamp_2 = maximum_2 - minimum_2
    # 周期分段
    x = np.arange(0, lens, step)  # start, stop, step
    # 分段最值及其索引
    intervals = [data[x[i]:x[i+1]] for i in range(len(x)-1)]
    min_idxs = np.zeros(len(intervals))
    min_vals = np.zeros(len(intervals))
    max_idxs = np.zeros(len(intervals))
    max_vals = np.zeros(len(intervals))
    for i, interval in enumerate(intervals):
        minidx = np.argmin(interval) + x[i]
        maxidx = np.argmax(interval) + x[i]
        minval = np.min(interval)
        maxval = np.max(interval)
        min_idxs[i] = minidx
        min_vals[i] = minval
        max_idxs[i] = maxidx
        max_vals[i] = maxval
    # 最值, 均值, 方差
    minimum = min(min_vals)
    maximum = max(max_vals)
    average = np.mean(data)
    standard = np.std(data)
    # 最值均值、最值均值振幅
    avgmin = np.mean([val for val in min_vals if (val > min(min_vals)) and (val < max(min_vals))])
    avgmax = np.mean([val for val in max_vals if (val > min(max_vals)) and (val < max(max_vals))])
    avgamp_1 = avgmax - avgmin
    avgamp = avgamp_1 if avgamp_1 >= avgamp_2 else avgamp_2
    # 振动异常
    diff_thr = avgamp * diff if diff < 1 else diff
    mindiff = abs(min_vals - avgmin)
    maxdiff = abs(max_vals - avgmax)
    minover = mindiff[mindiff > diff_thr]
    maxover = maxdiff[maxdiff > diff_thr]
    minover_count = len(minover)
    maxover_count = len(maxover)
    allover_count = minover_count + maxover_count

    # 返回结果
    result = {'max': maximum,
              'min': minimum,
              'avg': average,
              'std': standard,
              'avgMax': avgmax,
              'avgMin': avgmin,
              'avgAmp': avgamp,
              'maxOverCount': maxover_count,
              'minOverCount': minover_count,
              'allOverCount': allover_count}
    return result


def one_row(row: pd.Series, model_IF, kwargs={}):
    """检测一行数据"""
    amplitude = kwargs.get('amplitude', 6000)  # 正常振幅
    gzRatio = kwargs.get('gzRatio', 2)  # 共振倍数（正常振幅的倍数）
    pzRatio = kwargs.get('pzRatio', 1/2)  # 偏振比例（正常振幅的比例值）
    nzRatio = kwargs.get('nzRatio', 2)  # 扭振倍数（正常振幅的倍数）

    # 四点三轴时域加速度(l1,l2,r1,r2)
    xyz_time = row[[index for index in row.index if index[:6] in ['x_time', 'y_time', 'z_time']]]

    # 毛刺（异常毛刺）
    result = xyz_time.apply(one_cell, args=(kwargs,))
    for index in result.index:
        if result[index]['allOverCount'] > 0:
            result['maoci'] = True
            break
        else:
            result['maoci'] = False

    # 利用模型进行判断，数据有误异常（工作状态）
    result = one_time(xyz_time, kwargs, model_IF, result)

    # 0 存在异常
    # if result['isAbnormal'] == 0:  # 0 False, 1 True
    # 共振（振幅过大）
    gz_thr = amplitude * gzRatio
    if result['x_time_l1']['avgAmp'] > gz_thr and result['x_time_r1']['avgAmp'] > gz_thr:
        result['gongzhen'] = True
    else:
        result['gongzhen'] = False

    # 偏摆（产生了横向力）
    pb_thr = amplitude * pzRatio
    if result['z_time_l1']['avgAmp'] > pb_thr or result['z_time_r1']['avgAmp'] > pb_thr:
        result['pianzhen'] = True
    else:
        result['pianzhen'] = False

    # 扭振（垂直方向前后左右振幅差异大）
    l1amp = result['y_time_l1']['avgAmp']
    l2amp = result['y_time_l2']['avgAmp']
    r1amp = result['y_time_r1']['avgAmp']
    r2amp = result['y_time_r2']['avgAmp']
    if ((l1amp/r1amp > nzRatio) and (r2amp/l2amp > nzRatio)) or ((r1amp/l1amp > nzRatio) and (l2amp/r2amp > nzRatio)):
        result['niuzhen'] = True
    else:
        result['niuzhen'] = False
    # else:
    #     result['gongzhen'] = False
    #     result['pianzhen'] = False
    #     result['niuzhen'] = False

    result['time_l1'] = row['time_l1']
    result['time_l2'] = row['time_l2']
    result['time_r1'] = row['time_r1']
    result['time_r2'] = row['time_r2']
    result['time'] = row['time_l1']
    result['name'] = row['name_l1']
    result['pos'] = 'lr'
    result['mosun'] = False
    result['diantou'] = False
    return result


# def short_detect(device, kwargs={}):
#     """短期检测"""
#     last_row = device.iloc[-1]
#     result = one_row(last_row, kwargs)  # series
#     # result = dict(result)
#     # print(result)
#     # 振幅
#     # xamp_thr = (7000, 3000)  # 前后(max,min)
#     # yamp_thr = (7000, 3000)  # 上下(max,min)
#     # zamp_thr = (4000,)       # 左右(max)
#     return result


def middle_detect(device, model_IF, kwargs={}):
    """中期检测"""
    df = pd.DataFrame([])
    for i in range(len(device)):
        row = device.iloc[i]
        result = one_row(row, model_IF, kwargs)
        result = pd.DataFrame(dict(result.map(lambda x: [x])))
        df = pd.concat([df, result], axis=0)

    temp = df[[name for name in df.columns if name[:6] in ['x_time', 'y_time', 'z_time']]]
    maxdf = temp.applymap(lambda x: x['max'])
    mindf = temp.applymap(lambda x: x['min'])
    avgdf = temp.applymap(lambda x: x['avg'])
    avgmaxdf = temp.applymap(lambda x: x['avgMax'])
    avgmindf = temp.applymap(lambda x: x['avgMin'])
    avgmapdf = temp.applymap(lambda x: x['avgAmp'])
    maxovercountdf = temp.applymap(lambda x: x['maxOverCount'])
    minovercountdf = temp.applymap(lambda x: x['minOverCount'])

    result = dict()
    result['df'] = df
    result['maxDf'] = maxdf
    result['minDf'] = mindf
    result['avgDf'] = avgdf
    result['avgMaxDf'] = avgmaxdf
    result['avgMinDf'] = avgmindf
    result['avgMapDf'] = avgmapdf
    result['maxOverCountDf'] = maxovercountdf
    result['minOverCountDf'] = minovercountdf
    return result


def long_detect(device):
    """长期检测"""
    pass


def detect(cache_dict, result_dict, model_IF, kwargs={}):
    """检测异常"""
    for name in cache_dict.keys():
        device = cache_dict[name]
        if len(device) == 0:
            continue
        result = result_dict.get(name, pd.DataFrame({}))
        if len(result) > 0:
            last_time = result['time_l1'].iloc[-1]
            device = device[device['time_l1'] > last_time]  # avoid re-detect

        # 短期
        # result_short = short_detect(device, kwargs)  # 1 second
        # 中期
        result_middle = middle_detect(device, model_IF, kwargs)  # 1 hour
        # 长期
        # result_long = long_detect(device)  # 1 day

        # 整合
        result = pd.concat([result, result_middle['df']])
        result_dict[name] = result
        # break
    return result_dict


# ====================================================
#                        显示
# ====================================================
def draw(axes, loc, data, mode='plot', args={}):
    """绘图"""
    if isinstance(loc, int):
        if mode == 'plot':
            axes[loc].plot(data,
                           color=args.get('color'),
                           label=args.get('label'))
        elif mode == 'hist':
            axes[loc].hist(data, args['bins'],
                           color=args.get('color'),
                           label=args.get('label'))
        elif mode == 'scatter':
            axes[loc].scatter(data[0], data[1],
                              color=args.get('color'),
                              label=args.get('label'))
        elif mode == 'bar':
            axes[loc].bar(data[0], data[1],
                          color=args.get('color'),
                          label=args.get('label'))
        axes[loc].grid(alpha=0.5, linestyle='--')
    else:
        r, c = loc
        if mode == 'plot':
            axes[r, c].plot(data,
                            color=args.get('color'),
                            label=args.get('label'))
        elif mode == 'hist':
            axes[r, c].hist(data, args['bins'],
                            color=args.get('color'),
                            label=args.get('label'))
        elif mode == 'scatter':
            axes[r, c].scatter(data[0], data[1],
                               color=args.get('color'),
                               label=args.get('label'))
        elif mode == 'bar':
            axes[r, c].bar(data[0], data[1],
                           color=args.get('color'),
                           label=args.get('label'))
        axes[r, c].grid(alpha=0.5, linestyle='--')
