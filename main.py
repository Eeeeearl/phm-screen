import pandas as pd
import time
import json
import os
import sys

from detect_box import main as detect_box
from detect_shaker import main as detect_shaker
from utils.common import *
from config import *
from utils import model


class Detector(object):
    """检测器"""
    def __init__(self, kwargs={}):
        # 数据库参数字典
        self.db_kwargs = kwargs.get('db_kwargs')
        # 连接数据库
        self.engine, self.conn = connect_db2(self.db_kwargs)
        # 箱体参数字典
        self.box_kwargs = kwargs.get('box_kwargs')
        self.box_kwargs['conn'] = self.conn
        # 激振器参数字典
        self.shaker_kwargs = kwargs.get('shaker_kwargs')
        self.shaker_kwargs['conn'] = self.conn
        # ADD: BiLSTM模型加载
        self.model_olj = model.BiLSTM('./model/BiLSTM_pry_pre5.h5', pattern='olj')
        self.model_temp = model.BiLSTM('./model/BiLSTM_temp_pre5.h5', pattern='temp')
        self.model_IF = model.IForest('.//model/IForest.pkl')

    def __del__(self):
        # self.engine.close()
        self.conn.close()

    def update_kwargs(self, kwargs={}):
        for key in kwargs.keys():
            if key in self.box_kwargs.keys():
                self.box_kwargs[key] = kwargs[key]
            if key in self.shaker_kwargs.keys():
                self.shaker_kwargs[key] = kwargs[key]

    def run(self, kwargs={}):
        # 更新参数字典
        self.update_kwargs(kwargs)

        while True:
            # 箱体检测
            t0 = time.time()
            box_result = detect_box(self.box_kwargs, self.model_IF)
            print(f'--> 箱体检测完毕！{time.time()-t0:.3f}s')
            # print(dict(box_result['筛机3108'])['name'].iloc[-1])
            # print(box_result['筛机3108'].columns)

            # 激振器检测
            # ADD: add model、predict result
            t0 = time.time()
            shaker_result, pre_result = detect_shaker(self.shaker_kwargs, self.model_olj, self.model_temp)
            print(f'--> 激振器检测完毕！{time.time()-t0:.3f}s')
            # print(shaker_result)

            # 保存结果
            sql = 'select * from public.result order by time desc limit 1;'
            result = pd.read_sql(sql, self.conn)
            columns = result.columns.values
            # result is null
            if len(result) == 0:
                last_time = pd.to_datetime(0)
            else:
                last_time = result['time'].values[-1]
            for name in get_device_id_name(self.conn).values():
                # 保存筛箱结果
                if name in box_result.keys():
                    device = box_result[name]
                    device = device[device['time_l1'] > last_time]
                    if last_time == pd.to_datetime(0):
                        device = device[columns]
                        device = device.reset_index(drop=True)
                        ser = (device == True).any(axis=1)
                        index = ser[ser == True].index.values
                        device = device.iloc[index]
                        device.to_sql('result', self.engine, if_exists='append', index=False)
                    else:
                        for i in range(len(device)):
                            row = device.iloc[i]
                            t = row['time']
                            name = row['name']
                            pos = row['pos']
                            maoci = row['maoci']
                            gongzhen = row['gongzhen']
                            pianzhen = row['pianzhen']
                            niuzhen = row['niuzhen']
                            mosun = row['mosun']
                            diantou = row['diantou']
                            sql = f"""INSERT INTO public.result VALUES (
                                '{t}','{name}','{pos}',{maoci},{gongzhen},{pianzhen},{niuzhen},{mosun},{diantou}
                            );"""
                            if maoci or gongzhen or pianzhen or niuzhen or mosun or diantou:
                                self.conn.execute(text(sql))
                                self.conn.commit()
                # 保存激振器结果
                if name in shaker_result.keys():
                    device = shaker_result[name]
                    if len(device) == 0:
                        continue
                    device = device[device['time'] > last_time]
                    if last_time == pd.to_datetime(0):
                        device = device[columns]
                        device = device.reset_index(drop=True)
                        ser = (device == True).any(axis=1)
                        index = ser[ser == True].index.values
                        device = device.iloc[index]
                        device.to_sql('result', self.engine, if_exists='append', index=False)
                    for i in range(len(device)):
                        row = device.iloc[i]
                        t = row['time']
                        pos = row['pos']
                        maoci = row['maoci']
                        gongzhen = row['gongzhen']
                        pianzhen = row['pianzhen']
                        niuzhen = row['niuzhen']
                        mosun = row['mosun']
                        diantou = row['diantou']
                        sql = f"""INSERT INTO "public"."result" VALUES (
                            '{t}','{name}','{pos}',{maoci},{gongzhen},{pianzhen},{niuzhen},{mosun},{diantou}
                        );"""
                        if maoci or gongzhen or pianzhen or niuzhen or mosun or diantou:
                            self.conn.execute(text(sql))
                            self.conn.commit()
            # ADD: insert data
            if not pre_result.empty:
                try:
                    pre_result.to_sql('predict_yis', self.engine, if_exists='append', index=False)
                except Exception as e:
                    print(f"error info: {e}")
            print('--> 结果保存完毕！')
            print('-------------------------')

            # 检测频率：1min
            time.sleep(60)


if __name__ == '__main__':
    # # 生成json文件
    # json_dict = {
    #     'db_kwargs': db_kwargs,
    #     'box_kwargs': box_kwargs,
    #     'shaker_kwargs': shaker_kwargs
    # }
    # with open('./conf.json', 'w') as f:
    #     f.write(json.dumps(json_dict))

    # 读取json文件
    current_path = os.path.abspath(sys.argv[0])  # 主程序路经
    project_path = os.path.dirname(current_path)  # 上级目录（项目目录）
    file_path = os.path.join(project_path, 'conf.json')
    with open(file_path, 'r') as f:
        kwargs = json.load(f)

    # 创建实例
    det = Detector(kwargs)
    print('>>> 创建检测实例，数据库连接成功！')

    # 创建result表
    create_result(det.conn)
    print('>>> 创建result结果表！')
    # ADD: 创建predict表
    create_predict_yis(det.conn)
    print('>>> 创建predict预测表！')
    print('-------------------------')

    # 开始检测
    det.run()

    # 验证结果
    # print(pd.read_sql('select * from public.result;', det.conn))
