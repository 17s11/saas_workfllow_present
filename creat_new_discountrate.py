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


def get_new_discount_rating(query_config):           #新折算率生成函数
    logger.info('开始获取折算率原始数据')
    base_df = f'''SELECT td AS trade_date,stock_code,default_probability,conversion_rate,conversion_rate_adj FROM model_st
    WHERE td BETWEEN '{query_config['start_date']}' AND '{query_config['end_date']}' ;
    '''
    base_dfs=pd.read_sql(base_df,query_config['istock_conn'])
    ###########################最新HPE是错误的，在领鸟开发出HPE新字段后，记得替换HPE 字段！！！  2025.03.25##########################
    indicator_query = f'''SELECT td AS trade_date,stock_code,VOLATILITY_2S,hpe_new AS HPE,cct_adj_001 FROM indicator 
    WHERE td BETWEEN '{query_config['start_date']}' AND '{query_config['end_date']}';'''
    indicator_df=pd.read_sql(indicator_query,query_config['istock_conn'])      

    stock_info_query = '''SELECT * FROM stock_info;'''
    stock_info = pd.read_sql(stock_info_query, query_config['istock_conn'])

    query_rating = f'''SELECT stock_code,td AS trade_date,five_class_result_private_adj AS five_class_result_saas_adj FROM model_st 
    WHERE td BETWEEN '{query_config['start_date']}' AND '{query_config['end_date']}';'''
    rating_df=pd.read_sql(query_rating,query_config['istock_conn'])
    logger.info('折算率原始数据获取完毕')
    # def generate_ths_code_and_td_mkt(stock_code):
    #     if stock_code.startswith(('60', '68')):
    #         return f"{stock_code}.SH"
    #     elif stock_code.startswith(('00', '30')):
    #         return f"{stock_code}.SZ"
    #     else:
    #         return f"{stock_code}.BJ"
    # rating_df['stock_code']=rating_df['stock_code'].apply(generate_ths_code_and_td_mkt)
    rating_df['trade_date']=pd.to_datetime(rating_df['trade_date'])

    base_df=base_dfs.copy()

    # #### 新折算率计算
    logger.info('开始折算率规则判断！')
    index_list = stock_info[(stock_info['index_name'].str.contains('深证100', case=False, na=False)) | 
                            (stock_info['index_name'].str.contains('上证180', case=False, na=False))]['thscode'].tolist()
    # 在 base_df 中创建新列 rule_001，默认值为 NaN
    base_df['rule_001'] = 0.65
    # 更新 base_df 中 rule_001 列的值
    base_df.loc[base_df['stock_code'].isin(index_list), 'rule_001'] = 0.7     ### 上证180、深证100指数成份股股票折算率最高不超过70% and 其他A股股票折算率最高不超过65%

    drawback_list = indicator_df[indicator_df['VOLATILITY_2S']<0.3]['stock_code'].tolist()
    drawback_list = list(set(index_list).intersection(set(drawback_list)))
    base_df['rule_002'] = np.nan
    # 更新 base_df 中 rule_002 列的值
    base_df.loc[base_df['stock_code'].isin(drawback_list), 'rule_002'] = 0.7   ###3个月最大回撤小于0.3且为上证180、深证100指数成份股股票折算率调至70%

    def create_timepoint_list(query_config,date_group):     ###计算时间点列表（上市5天/3个月/一年）
        new_1_query = f'''SELECT MIN(td_date) AS earliest_date
        FROM (
            SELECT DISTINCT td_date
            FROM shsz_stock_daily_quotation ssdq 
            WHERE td_date < '{date_group}'
            ORDER BY td_date DESC
            LIMIT 5
        ) AS recent_dates;'''
        new_1 = pd.read_sql(new_1_query, query_config['istock_conn'])
        new_1 = new_1.iloc[0][0]
        new_2 = pd.to_datetime(date_group) - relativedelta(months=3)
        new_3 = pd.to_datetime(date_group) - relativedelta(months=12)
        timepoint_list = [new_1,new_2,new_3]
        return timepoint_list




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

caluculate_discount_rate()