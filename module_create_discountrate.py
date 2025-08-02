import sqlalchemy as sa
from sqlalchemy import create_engine, text
import pymysql
import sqlite3

import numpy as np
import pandas as pd
from datetime import datetime, timedelta,date
import seaborn as sns

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from data_logger import setup_logger
import module_query_update


import warnings
import os
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

logger = setup_logger()



def init(start_date=None, end_date=None):    
    config = {
        "start_date": start_date,
        "end_date": end_date,
        "istock_conn": getConnection("istock"),
        "jqka_conn": getConnection("jqka"),
        "rating_conn" : create_engine('sqlite:///result/rating_db.db')  ### 记得改文件路径！！！
    }
    logger.info(f"折算率计算初始化完成！查询时间段为：{start_date} 至 {end_date}")
    return config


def get_new_discount_rating(query_config):           #新折算率生成函数
   
        return timepoint_list

    def create_rule_003456(base_df_group,stock_info,query_config):

    return base_df

def save_to_db(base_df, query_config):
    engine = query_config['rating_conn']
    error_occurred = False  # 添加一个标志变量用于检查是否发生错误
    BATCH_SIZE = 1000  # 设置批次大小

    with engine.connect() as conn:    
        for i in range(0, len(base_df), BATCH_SIZE):
            batch_df = base_df.iloc[i:i + BATCH_SIZE]
            
            # 将DataFrame转换为字典列表，并将NaN值替换为None
            records = [
                {
                    'trade_date': None if pd.isna(row[0]) else row[0],
                    'stock_code': None if pd.isna(row[1]) else row[1],
                    'default_probability': None if pd.isna(row[2]) else row[2],
                    'conversion_rate': None if pd.isna(row[3]) else row[3],
                    'conversion_rate_adj': None if pd.isna(row[4]) else row[4],
                    'rule_001': None if pd.isna(row[5]) else row[5],
                    'rule_002': None if pd.isna(row[6]) else row[6],
                    'rule_003': None if pd.isna(row[7]) else row[7],
                    'rule_004': None if pd.isna(row[8]) else row[8],
                    'rule_005': None if pd.isna(row[9]) else row[9],
                    'rule_006': None if pd.isna(row[10]) else row[10],
                    'rule_007': None if pd.isna(row[11]) else row[11],
                    'rule_008': None if pd.isna(row[12]) else row[12],
                    'base_line_1': None if pd.isna(row[13]) else row[13],
                    'rating_weight': None if pd.isna(row[14]) else row[14],
                    'base_line_2': None if pd.isna(row[15]) else row[15],
                    'fin_conversion_rate': None if pd.isna(row[16]) else row[16],
                    'five_class_result_saas_adj': None if pd.isna(row[17]) else row[17]
                }
                for row in batch_df.itertuples(index=False)
            ]

            insert_query = text("""
            INSERT OR IGNORE INTO new_discount_record 
            (trade_date, stock_code, default_probability, conversion_rate, conversion_rate_adj, rule_001, rule_002, rule_003, rule_004, rule_005, rule_006, rule_007, rule_008, base_line_1, rating_weight, base_line_2, fin_conversion_rate, five_class_result_saas_adj) 
            VALUES (:trade_date, :stock_code, :default_probability, :conversion_rate, :conversion_rate_adj, :rule_001, :rule_002, :rule_003, :rule_004, :rule_005, :rule_006, :rule_007, :rule_008, :base_line_1, :rating_weight, :base_line_2, :fin_conversion_rate, :five_class_result_saas_adj)
            """)

            try:
                # 使用executemany进行批量插入
                result = conn.execute(insert_query, records)
                # 提交事务
                conn.commit()
                
            except Exception as e:
                logger.info(f"Error inserting rows for batch starting at index {i}: {e}")
                error_occurred = True
                break  # 发生错误立即退出
                
        if not error_occurred:
            logger.info("base_df数据已成功导入到result/rating_db.new_discount_record!")
        else:
            logger.info("由于错误，折算率数据插入被中止。")


def caluculate_discount_rate():
    query_config=init(start_date=None, end_date=None)
    end_date = pd.read_sql("SELECT MAX(td) FROM model_st ms ;",query_config['istock_conn'])
    end_date=end_date.loc[0][0].strftime('%Y-%m-%d')
    new_date=pd.read_sql('SELECT MAX(trade_date) FROM new_discount_record;',query_config['rating_conn'])
    new_date=new_date.loc[0][0]
    if pd.to_datetime(new_date)==pd.to_datetime(end_date):
        logger.info('no new discount data')  
    else: 
        start_date=pd.read_sql(f'''SELECT MIN(td)  FROM model_st
                                WHERE td > '{new_date}'; ''',query_config['istock_conn'])
        start_date=start_date.loc[0][0].strftime('%Y-%m-%d')
        query_config=init(start_date=start_date, end_date=end_date)
        module_query_update.daily_query_update()
        base_df = get_new_discount_rating(query_config)
        save_to_db(base_df,query_config)


def caluculate_discount_rate_history(s,e):
    query_config=init(start_date=s, end_date=e)
    base_df = get_new_discount_rating(query_config)
    return base_df


def caluculate_discount_rate_initialize(range_date):
    query_config=init(start_date=None, end_date=None)
    end_date = pd.read_sql("SELECT MAX(td) FROM model_st ms ;",query_config['istock_conn'])
    end_date=end_date.loc[0][0].strftime('%Y-%m-%d')
    new_date=range_date
    start_date=pd.read_sql(f'''SELECT MIN(td)  FROM model_st
                            WHERE td > '{new_date}'; ''',query_config['istock_conn'])
    start_date=start_date.loc[0][0].strftime('%Y-%m-%d')
    query_config=init(start_date=start_date, end_date=end_date)
    module_query_update.daily_query_update()
    base_df = get_new_discount_rating(query_config)
    save_to_db(base_df,query_config)