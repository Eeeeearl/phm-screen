
# 数据库参数字典
db_kwargs = {
    'database': 'postgres',
    'user': 'postgres',
    'password': 'postgres',
    'host': '127.0.0.1',
    'port': 5432,
}

# 箱体参数字典
box_kwargs = {
    # read data
    'conn': None,
    'start_time': '',
    'end_time': '',
    'timedelta': 90,     # 过去30天数据
    'room': 4*24*90,     # 缓存容量
    # one_cell
    'step': 1000 // 15,  # 步长：1个周期长度（1000：数据量，15：周期数）
    'lens': 1024,        # 传感器采样频率（1s采样数据量）
    'diff': 1 / 2,       # 若传值<1则计算振幅比例值，若传值>1则使用该值
    # one_row
    'amplitude': 6000,   # 正常振幅
    'gzRatio': 2,        # 共振倍数（正常振幅的倍数）
    'pzRatio': 1 / 2,    # 偏振比例（正常振幅的比例值）
    'nzRatio': 2,        # 扭振倍数（正常振幅的倍数）
    # one_time
    'outRatio': 0.2,     # 离群点比例（若一组数据超过正常数量的20%，则说明存在异常）
    'minData': 800,      # 单条数据最少数量（正常1024）
}

# 激振器参数字典
shaker_kwargs = {
    'conn': None,
    'start_time': '',
    'end_time': '',
    'daydelta': 1,              # 1 day
    'az_range': [-34, 48],      # x轴加速度范围
    'ay_range': [-6, 6],        # y轴加速度范围
    'ax_range': [-15, 3],       # z轴加速度范围
    'mul': 1.5,                 # 共振系数
    'pitch_range': [-90, 90],   # 俯仰角范围
    'roll_range': [-180, 180],  # 翻滚角范围
    'yaw_range': [-180, 180],   # 偏航角范围
    'temp_range': [0, 55],      # 正常工作温度范围
    'temp': None,               # 单位时间内最大温差，单位：°
    'time': 30,                 # 单位时间间隔，单位：min
    'nor_interval': 60,         # 默认采集数据间隔时间，单位：second
}
