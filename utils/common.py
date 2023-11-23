import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text


def connect_db(kwargs=dict()):
    """连接数据库"""
    database = kwargs.get('database', "postgres")
    user = kwargs.get('user', 'postgres')
    password = kwargs.get('password', 'postgres')
    host = kwargs.get('host', "127.0.0.1")
    port = kwargs.get('port', 5432)
    conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
    return conn


def connect_db2(kwargs=dict()):
    """连接数据库"""
    database = kwargs.get('database', "postgres")
    user = kwargs.get('user', 'postgres')
    password = kwargs.get('password', 'postgres')
    host = kwargs.get('host', "127.0.0.1")
    port = kwargs.get('port', 5432)
    engine = create_engine('postgresql://{}:{}@{}:{}/{}?'.format(user, password, host, port, database), future=True)
    conn = engine.connect()
    return engine, conn


def get_device_id_name(conn):
    sql = "select id, name from public.device where deleted=0;"
    df = pd.read_sql(sql, conn)
    return df.set_index('id')['name'].to_dict()


def create_result(conn):
    """创建结果保存表"""
    # conn.execute(text('DROP TABLE IF EXISTS public.result;'))
    conn.execute(text(
        """
            CREATE TABLE IF NOT EXISTS public.result (
                "time" timestamp(6) NOT NULL,
                "name" varchar(20)  NOT NULL,
                "pos" varchar(20) NOT NULL,
                "maoci" boolean NOT NULL,
                "gongzhen" boolean NOT NULL,
                "pianzhen" boolean NOT NULL,
                "niuzhen" boolean NOT NULL,
                "mosun" boolean NOT NULL,
                "diantou" boolean NOT NULL
            );
        """
    ))
    conn.commit()


# ADD: create predict yis data table
def create_predict_yis(conn):
    """创建结果保存表
    字段：
    time：预测时间点后一分钟的时间（第一个预测数据的时间）
    """
    # conn.execute(text('DROP TABLE IF EXISTS public.result;'))
    conn.execute(text(
        """
            CREATE TABLE IF NOT EXISTS public.predict_yis (
                "time" timestamp(6) without time zone NOT NULL,
                code character varying(20),
                ax numeric[],
                ay numeric[],
                az numeric[],
                wx numeric[],
                wy numeric[],
                wz numeric[],
                pitch numeric[],
                roll numeric[],
                yaw numeric[],
                q0 numeric[],
                q1 numeric[],
                q2 numeric[],
                q3 numeric[],
                temperature numeric[],
                device_id character varying(20),
                pos character varying(10),
                PRIMARY KEY (time)
            );
        """
    ))
    conn.commit()