# 入库与初始化函数
import sqlalchemy as sa
import pymysql
import sqlite3

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from datetime import date

from data_logger import setup_logger
import warnings
import os
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
from sqlalchemy import create_engine, text

logger = setup_logger()

def init(start_date=None, end_date=None):    
    config = {
        "start_date": start_date,
        "end_date": end_date,
        "istock_conn": getConnection("istock"),
        "jqka_conn": getConnection("jqka"),
        "rating_conn" : create_engine('sqlite:///result/rating_db.db')
    }
    logger.info(f"五级分类生成程序初始化完成！查询时间段为：{start_date} 至 {end_date}")
    return config

def execute_with_error_handling(func, query_config, func_name):    #取数报错日志
    try:
        return func(query_config)
    except Exception as e:
        logger.info(f"取数函数 {func_name} 运行时发生错误：{e}")

def empty_folder(folder_path):   #清空文件夹
    """
    清空指定文件夹的内容（包括子文件夹及其内容），但保留该文件夹本身。
    
    :param folder_path: 要清空的文件夹路径
    """
    if not os.path.exists(folder_path):
        logger.info(f"文件夹 {folder_path} 不存在")
        return
    
    for root, dirs, files in os.walk(folder_path, topdown=False):
        # 删除所有文件
        for name in files:
            file_path = os.path.join(root, name)
            try:
                os.remove(file_path)  # 删除文件
            except Exception as e:
                logger.info(f'Failed to delete file {file_path}. Reason: {e}')
        
        # 删除所有子文件夹
        for name in dirs:
            dir_path = os.path.join(root, name)
            try:
                os.rmdir(dir_path)  # 删除空目录
            except Exception as e:
                logger.info(f'Failed to delete directory {dir_path}. Reason: {e}')

def insert_data_without_duplicates(df, table_name):
    engine = create_engine('sqlite:///result/rating_db.db')
    
    with engine.connect() as connection:
        transaction = connection.begin()
        for idx, row in df.iterrows():
            columns = ', '.join(df.columns)
            placeholders = ', '.join([':' + str(i) for i in range(len(df.columns))])
            values = {str(i): value for i, value in enumerate(row)}
            insert_stmt = text(f"""
                INSERT OR IGNORE INTO {table_name} ({columns})
                VALUES ({placeholders})
            """)
            try:
                connection.execute(insert_stmt, values)
            except Exception as e:
                logger.info(f"Error inserting row {row}: {e}")
                break
        transaction.commit()
    logger.info(table_name+'入库')




# 取数程序

def get_market_value(config):
    engine_istock = config["istock_conn"]
    start_date = config["start_date"]
    end_date = config["end_date"]      ###此处的sql中的open_price时收盘价而非真正的open_price,未作该是因为防止报错
    sql_quote_query = f"""
    SELECT 
    td_date,
    sec_code,
    sec_short_name,
    pre_close * (1 + COALESCE(chg_ratio, 0) / 100) AS open_price     
    FROM 
    shsz_stock_daily_quotation
    WHERE 
    td_date BETWEEN DATE_SUB('{start_date}', INTERVAL 90 DAY) AND '{end_date}'
    AND sec_category = 'A股'
    """
    price_df=pd.read_sql(sql_quote_query,engine_istock)  #读取价格数据


    share_query = f'''SELECT css.publish_date,sbi.sec_code,css.float_shares
        FROM corp_share_structure css 
        JOIN sec_basic_info sbi ON css.corp_code=sbi.issue_org_id
        WHERE sbi.sec_type='A股'; '''
    share_df=pd.read_sql(share_query,engine_istock)    #读取流通股本数据
    share_df['publish_date']=pd.to_datetime(share_df['publish_date'])

    def share_select(group,start_date):
        before_3_month = datetime.strptime(start_date, "%Y-%m-%d") -timedelta(days=90)
        if before_3_month>=group['publish_date'].max():
            if len(group.loc[group['publish_date']==group['publish_date'].max()])>1:
                return group.loc[group['publish_date']==group['publish_date'].max()].tail(1)
            else:
                return group.loc[group['publish_date']==group['publish_date'].max()]
        else:
            group = group.loc[group['publish_date']>=group.loc[group['publish_date']<before_3_month]['publish_date'].max()]
            return group.groupby('publish_date').apply(lambda x:x.tail(1)).reset_index(drop=True)
        
    def add_float_shares(group,share_df):
        if len(share_df.loc[share_df['sec_code']==group['sec_code'].unique()[0]])==1:
            group['float_shares']=share_df.loc[share_df['sec_code']==group['sec_code'].unique()[0]]['float_shares'].values[0]
        else:
            def update_float_shares(row,share_info):
                a=share_info.loc[share_info['publish_date']<=row['td_date']]
                return a.loc[a['publish_date']==a['publish_date'].max()]['float_shares'].values[0]
            stock_info=share_df.loc[share_df['sec_code']==group['sec_code'].unique()[0]]
            group['float_shares']=group.apply(lambda row:update_float_shares(row,stock_info),axis=1)
        return group

    share_df=share_df.groupby('sec_code').apply(share_select,start_date=start_date).reset_index(drop=True)
    market_value_df=price_df.groupby('sec_code').apply(add_float_shares,share_df=share_df).reset_index(drop=True)

    market_value_df['float_market_value']=market_value_df['float_shares']*market_value_df['open_price']  # 计算流通市值
    market_value_df.sort_values(by=['sec_code', 'td_date'], inplace=True)
    market_value_df.set_index('td_date', inplace=True)
    # 对每个 'sec_code' 分别计算滚动平均值
    market_value_df['rolling_avg_float_market_value'] = market_value_df.groupby('sec_code')['float_market_value'].transform(lambda x: x.rolling(window='90D').mean())
    market_value_df.reset_index(inplace=True)
    market_value_df=market_value_df.loc[market_value_df['td_date']>=start_date]
    market_value_df=market_value_df[['td_date','sec_code','sec_short_name','rolling_avg_float_market_value']]    # 输出流通市值平均dataframe
    market_value_df['td_date'] = market_value_df['td_date'].dt.strftime('%Y-%m-%d')
    market_value_df.columns=['trade_date','stock_code','sec_short_name','rolling_avg_float_market_value']
    market_value_df.to_csv('temp_model_backtesting/market_value.csv',index=False) # 输出流通市值平均dataframe
    return logger.info('流通市值计算完成,临时存入temp_model_backtesting/market_value.csv')

def get_goodwill_ratio(config):    # 获取商誉函数
        engine_istock = config["istock_conn"]
        start_date = config["start_date"]
        end_date = config["end_date"]
        sql_goodwill_ratio = f"""
                SELECT stock_code AS stock_code, td AS trade_date, PROFITABILITY_4 AS goodwill_ratio
                FROM indicator
                WHERE td BETWEEN '{start_date}' AND '{end_date}'
                """
        goodwill_ratio_df = pd.read_sql(sql_goodwill_ratio, engine_istock)
        goodwill_ratio_df['stock_code'] = goodwill_ratio_df['stock_code'].str[:6]
        goodwill_ratio_df['goodwill_ratio'].fillna(0, inplace=True)
        goodwill_ratio_df['stock_code']=goodwill_ratio_df['stock_code'].astype(str)
        goodwill_ratio_df.to_csv("temp_model_backtesting/goodwill_ratio.csv", index=False)
        del goodwill_ratio_df
        return  logger.info("商誉占比计算完毕,临时存入temp_model_backtesting/goodwill_ratio.csv")

def get_trans_amt(config):
        engine_istock = config["istock_conn"]
        start_date = config["start_date"]
        end_date =  config["end_date"]

        sql_trans_amt = f"""
                        SELECT stock_code  , td AS trade_date, SPJ_001 AS trans_amt,CJL_001 AS trans_vol
                        FROM evidence_daily
                        WHERE td BETWEEN DATE_SUB('{start_date}', INTERVAL 90 DAY) AND '{end_date}'
                        """
        df_trans_amt = pd.read_sql(sql_trans_amt, engine_istock)
        df_trans_amt['trans_amt']=df_trans_amt['trans_amt']*df_trans_amt['trans_vol']    #计算成交额
        # 确保日期是datetime格式，并仅保留年月日部分
        df_trans_amt['trade_date'] = pd.to_datetime(df_trans_amt['trade_date'])
        df_trans_amt.set_index('trade_date', inplace=True)
        # 计算每个股票的交易额的滚动平均值
        df_trans_amt['rolling_avg_trans_amt'] = df_trans_amt.groupby('stock_code')['trans_amt'].transform(lambda x: x.rolling(window='90D').mean())
        df_trans_amt.reset_index(inplace=True)
        df_trans_amt=df_trans_amt.loc[df_trans_amt['trade_date']>=start_date]
        df_trans_amt['td_date'] = df_trans_amt['trade_date'].dt.strftime('%Y-%m-%d')
        df_trans_amt['stock_code'] = df_trans_amt['stock_code'].str[:6]
        df_trans_amt=df_trans_amt[['trade_date','stock_code','rolling_avg_trans_amt']]
        df_trans_amt['stock_code']=df_trans_amt['stock_code'].astype(str)
        df_trans_amt=df_trans_amt[['stock_code','trade_date','rolling_avg_trans_amt']]
        df_trans_amt.to_csv('temp_model_backtesting/trans_amt.csv',index=False)
        del df_trans_amt
        return logger.info('日均交易额计算完毕,临时存入temp_model_backtesting/trans_amt.csv')

def convert_percentage_to_float(series):     ##数据处理函数，用于get_dicount_rate的字符串字段
    def convert_value(value):
        if isinstance(value, str) and value.endswith('%'):
            return float(value.rstrip('%')) / 100.0
        elif isinstance(value, str):
            if float(value)>1:
                return float(value)/100
            else:
                return float(value)
        else:
            raise ValueError(f"Unexpected value type: {type(value)}") 
    return series.apply(convert_value)

def get_discount_rate(config):     #同业折算率孰低值获取
    engine_istock = config["istock_conn"]
    start_date = config["start_date"]
    end_date = config["end_date"]

    # 将 start_date 转换为日期格式
    start_date_date = datetime.strptime(start_date, '%Y-%m-%d')
    comparison_date = datetime(2024, 10, 9)

    # 判断 start_date 是否早于 2024-10-09
    if start_date_date < comparison_date:
        # 创建一个空 DataFrame 并保存为 CSV 文件
        df_empty = pd.DataFrame(columns=['trade_date', 'stock_code', 'discount_rate'])
        df_empty.to_csv('temp_model_backtesting/discount_rate.csv', index=False)
        logger.info("由于开始日期早于2024-10-09,已生成空CSV文件")
    else:
        sql_discount_rate = f"""
        SELECT m.broker_name, m.stock_code , m.stock_name, m.tra_date AS trade_date, m.discount_rate
            FROM margin_trading_records m
            WHERE 
                m.tra_date BETWEEN '{start_date}' AND '{end_date}'
                AND m.broker_name IN ('中信证券', '中信建投', '招商证券', '广发证券', '平安证券', '华泰证券', '国信证券', '国泰君安')
                AND m.discount_rate IS NOT NULL
                AND m.concentration REGEXP '^[^[:space:]]+$'
            """
        # 从数据库中读取数据
        df_discount_rate = pd.read_sql(sql_discount_rate, engine_istock)
        df_discount_rate['trade_date'] = pd.to_datetime(df_discount_rate['trade_date'])

        # 假设 convert_percentage_to_float 是已经定义好的函数
        df_discount_rate['discount_rate'] = convert_percentage_to_float(df_discount_rate['discount_rate'])

        df_discount_rate = df_discount_rate.groupby(['trade_date', 'stock_code']).agg({
            'stock_name': 'first',
            'discount_rate': 'min'}).reset_index()   #取同业最低值
        df_discount_rate = df_discount_rate[['trade_date','stock_code','discount_rate']]
        df_discount_rate['trade_date'] = df_discount_rate['trade_date'].dt.strftime('%Y-%m-%d')
        df_discount_rate['stock_code'] = df_discount_rate['stock_code'].astype(str) 

        df_discount_rate.to_csv('temp_model_backtesting/discount_rate.csv', index=False)
        del df_discount_rate
        logger.info('孰低折算率已经获取完毕,临时存入temp_model_backtesting/discount_rate.csv')


##################################################################################################
########################################这里要改!!!!!##############################################
def get_collateral_ratio(config):     #抵押比例获取函数
        engine_istock = config["istock_conn"]
        start_date = config["start_date"]
        end_date =  config["end_date"]
        if start_date==end_date:
            sql_collateral_ratio = f"""
                        SELECT mcr.stock_code AS sec_code, 
                latest_dates.latest_date AS trade_date, 
                mcr.ratio AS collateral_ratio, 
                mcr.stock_name AS sec_short_name_cn
            FROM market_collateral_records mcr
            JOIN (
                SELECT stock_code, MAX(date) AS latest_date
                FROM market_collateral_records
                GROUP BY stock_code
            ) latest_dates ON mcr.stock_code = latest_dates.stock_code AND mcr.date = latest_dates.latest_date;
                    """
            df_collateral_ratio = pd.read_sql(sql_collateral_ratio, engine_istock)
            df_collateral_ratio['trade_date']=end_date
        else:
            sql_collateral_ratio_1=f'''SELECT mcr.stock_code AS sec_code, mcr.date AS trade_date, mcr.ratio AS collateral_ratio, mcr.stock_name AS sec_short_name_cn
                        FROM market_collateral_records mcr 
                        WHERE mcr.date >= '{start_date}' AND mcr.date < '{end_date}';'''
            df_collateral_ratio_1 = pd.read_sql(sql_collateral_ratio_1, engine_istock)
            sql_collateral_ratio_2=f"""
                        SELECT mcr.stock_code AS sec_code, 
                latest_dates.latest_date AS trade_date, 
                mcr.ratio AS collateral_ratio, 
                mcr.stock_name AS sec_short_name_cn
            FROM market_collateral_records mcr
            JOIN (
                SELECT stock_code, MAX(date) AS latest_date
                FROM market_collateral_records
                GROUP BY stock_code
            ) latest_dates ON mcr.stock_code = latest_dates.stock_code AND mcr.date = latest_dates.latest_date;
                    """
            df_collateral_ratio_2 = pd.read_sql(sql_collateral_ratio_2, engine_istock)
            df_collateral_ratio_2['trade_date']=end_date
            df_collateral_ratio=pd.concat([df_collateral_ratio_1,df_collateral_ratio_2],axis=0)


                # 将 sec_code 映射到 sec_basic_info 表中的 thscode，命名为 stock_code
        sql_sec_mapping = """
                SELECT sbi.sec_code, sbi.thscode AS stock_code, sbi.sec_short_name_cn
                FROM sec_basic_info sbi
                """
        df_sec_mapping = pd.read_sql(sql_sec_mapping, engine_istock)

                # 合并映射表，将 sec_code 映射为 stock_code
        df_collateral_ratio = df_collateral_ratio.merge(df_sec_mapping, on=['sec_code','sec_short_name_cn'], how='inner')
        df_collateral_ratio = df_collateral_ratio[['sec_code', 'trade_date', 'collateral_ratio']]
        df_collateral_ratio.columns = ['stock_code', 'trade_date', 'collateral_ratio']
        df_collateral_ratio['trade_date']=df_collateral_ratio['trade_date'].astype(str)
        df_collateral_ratio['stock_code']=df_collateral_ratio['stock_code'].astype(str)
        df_collateral_ratio.to_csv('temp_model_backtesting\collateral_ratio.csv',index=False)
        del df_collateral_ratio
        del df_sec_mapping
        return logger.info('单一担保物质押比例已经获取完毕,临时存入temp_model_backtesting/collateral_ratio.csv')

def get_delloitte_rating(config):     #德勤评级获取函数(目前弃用)
        engine_istock = config["istock_conn"]
        start_date = config["start_date"]
        end_date =  config["end_date"]

        sql_deloitte_rating = f"""
                SELECT stock_code AS stock_code, td AS trade_date, five_class_result_adj
                FROM model_st
                WHERE td BETWEEN '{start_date}' AND '{end_date}'
                """

        df_deloitte_rating = pd.read_sql(sql_deloitte_rating, engine_istock)    #获取德勤评级
        df_deloitte_rating['stock_code'] = df_deloitte_rating['stock_code'].str[:6]
        df_deloitte_rating['stock_code'] = df_deloitte_rating['stock_code'].astype(str)
        df_deloitte_rating['trade_date'] = df_deloitte_rating['trade_date'].astype(str)
        df_deloitte_rating.to_csv('temp_model_backtesting/deloitte_rating.csv', index=False)
        del df_deloitte_rating
        return logger.info("德勤评级数据获取成功,临时保存在 temp_model_backtesting/deloitte_rating.csv")

def get_deloitte_rating_1(config):
    # 获取德勤评级数据(smooth_score)
    engine_istock = config["istock_conn"]
    start_date = config["start_date"]
    end_date = config["end_date"]

    sql_deloitte_rating = f"""SELECT stock_code AS stock_code, td AS trade_date, smooth_score AS smooth_score 
    FROM model_st WHERE td BETWEEN '{start_date}' AND '{end_date}'
    """

    df_deloitte_rating = pd.read_sql(sql_deloitte_rating, engine_istock)  # 获取smooth_score
    df_deloitte_rating['stock_code'] = df_deloitte_rating['stock_code'].str[:6]
    df_deloitte_rating['stock_code'] = df_deloitte_rating['stock_code'].astype(str)
    df_deloitte_rating['trade_date'] = pd.to_datetime(df_deloitte_rating['trade_date'])

    # 按 trade_date 分组并计算每日的百分位数阈值


    def calculate_daily_thresholds(group):
        thresholds = group['smooth_score'].quantile(percentiles).tolist()
        return thresholds

    daily_thresholds = df_deloitte_rating.groupby('trade_date').apply(calculate_daily_thresholds)

    # 定义分类函数
    def classify(score, date):
        thresholds = daily_thresholds[date]
        if score >= thresholds[0]:
            return 'A'
        elif score >= thresholds[1]:
            return 'B'
        elif score >= thresholds[2]:
            return 'C'
        elif score >= thresholds[3]:
            return 'D'
        else:
            return 'E'

    # 应用分类函数
    df_deloitte_rating['five_class_result_adj'] = df_deloitte_rating.apply(
        lambda row: classify(row['smooth_score'], row['trade_date']), axis=1
    )

    # 选择需要的列
    df_deloitte_rating = df_deloitte_rating[['stock_code', 'trade_date', 'five_class_result_adj']]
    # 保存到CSV文件
    df_deloitte_rating.to_csv('temp_model_backtesting/deloitte_rating_1.csv', index=False)
    
    del df_deloitte_rating
    return logger.info("德勤评级数据_1获取成功,临时保存在 temp_model_backtesting/deloitte_rating_1.csv")

def get_stock_info(config):       #股票基本信息获取（基础股票池）
    sql='''SELECT thscode AS stock_code,listed_board_name ,listed_date ,td_mkt  ,st_issuance ,index_name FROM stock_info;'''
    df = pd.read_sql(sql,config["istock_conn"])

    df['stock_code']=df['stock_code'].astype(str)
    df['listed_date']=df['listed_date'].astype(str)
    ################当前版本，此处暂时不用（无需index_name 和 index_name_1）###################
    x=pd.read_excel(r'password_attention\中证800成分股数据.xlsx')
    xp=x['\t股票代码'].str[1:10].values
    df['index_name_1'] = df['stock_code'].apply(lambda x: '中证800' if x in xp else None)   #填入中证800字段
    #########################################################################################
    df['stock_code'] = df['stock_code'].str[:6]
    #df['st_issuance'].fillna('注册制',inplace=True)
    df.to_csv('temp_model_backtesting/stock_info.csv',index=False)
    del df
    return logger.info('获取最新stock_info成功,临时存在temp_model_backtesting/stock_info.csv')

def wanlian_data_merge():    #合并函数
    directory_path='temp_model_backtesting'
    try:    
        all_items = os.listdir(directory_path)
        files = [item for item in all_items if os.path.isfile(os.path.join(directory_path, item))]
    except Exception as e:
        logger.info(f"读取临时文件名发生了一个错误：{e}")

    wanlian_df=None
    try:
        trans_amt=pd.read_csv(directory_path+'/'+files[6],dtype={'stock_code': str, 'trade_date': str})
        market_value=pd.read_csv(directory_path+'/'+files[4],dtype={'stock_code': str, 'trade_date': str})
        wanlian_df=pd.merge(trans_amt,market_value,on=['trade_date','stock_code'],how='left')
        del trans_amt
        del market_value
        collateral_ratio=pd.read_csv(directory_path+'/'+files[0],dtype={'stock_code': str, 'trade_date': str})
        wanlian_df=pd.merge(wanlian_df,collateral_ratio,on=['trade_date','stock_code'],how='left')
        del collateral_ratio
        delloitte_rating=pd.read_csv(directory_path+'/'+files[1],dtype={'stock_code': str, 'trade_date': str})
        wanlian_df=pd.merge(wanlian_df,delloitte_rating,on=['trade_date','stock_code'],how='left')
        del delloitte_rating
        discount_rate=pd.read_csv(directory_path+'/'+files[2],dtype={'stock_code': str, 'trade_date': str})
        if len(discount_rate)>2:
            wanlian_df=pd.merge(wanlian_df,discount_rate,on=['trade_date','stock_code'],how='left')
            del discount_rate
        else:
            logger.info('############## discount_rate数据为空,忽略此条件 ##############')
        goodwill_ratio=pd.read_csv(directory_path+'/'+files[3],dtype={'stock_code': str, 'trade_date': str})
        wanlian_df=pd.merge(wanlian_df,goodwill_ratio,on=['trade_date','stock_code'],how='left')
        del goodwill_ratio
        stock_info=pd.read_csv(directory_path+'/'+files[5],dtype={'stock_code': str, 'trade_date': str})
        wanlian_df=pd.merge(wanlian_df,stock_info,on=['stock_code'],how='left')
        del stock_info
        wanlian_df['rolling_avg_trans_amt_rank']=wanlian_df.groupby('trade_date')['rolling_avg_trans_amt'].rank(pct=True, ascending=False)
        wanlian_df['rolling_avg_float_market_value_rank']=wanlian_df.groupby('trade_date')['rolling_avg_float_market_value'].rank(pct=True, ascending=False)
        logger.info('万联数据已准备好！')
        return wanlian_df
    except Exception as e:
        logger.info(f"合并表时发生了一个错误：{e}")

def get_saas_data(query_config):
    # 内置functions_to_execute列表
    functions_to_execute = [
        (get_market_value, "get_market_value"),
        (get_goodwill_ratio, "get_goodwill_ratio"),
        (get_trans_amt, "get_trans_amt"),
        (get_discount_rate, "get_discount_rate"),
        (get_collateral_ratio, "get_collateral_ratio"),
        #(get_delloitte_rating, "get_delloitte_rating"),  # 注释掉的示例行
        (get_deloitte_rating_1, "get_deloitte_rating_1"),
        (get_stock_info, "get_stock_info")
    ]
    # 执行所有取数函数并处理可能的异常
    for func, name in functions_to_execute:
        execute_with_error_handling(func, query_config, name)

    # 合并所有取数结果为一个wanlian_df
    wanlian_df = wanlian_data_merge()

    wanlian_df_test = wanlian_df.copy()
    insert_data_without_duplicates(wanlian_df, 'saas_indicator')
    del wanlian_df
    
    wanlian_df_test['index_name'].fillna("", inplace=True)
    
    default_collateral_ratio = 0
    wanlian_df_test['collateral_ratio'] = wanlian_df_test.groupby('stock_code')['collateral_ratio']\
        .transform(lambda x: x.fillna(x.mean()) if x.notnull().any() else default_collateral_ratio)
    
    if 'discount_rate' in wanlian_df_test.columns:
        default_discount_rate = 0.8
        wanlian_df_test['discount_rate'] = wanlian_df_test.groupby('stock_code')['discount_rate']\
                .transform(lambda x: x.fillna(x.mean()) if x.notnull().any() else default_discount_rate)
    
    wanlian_df_test['trade_date'] = pd.to_datetime(wanlian_df_test['trade_date'], errors='coerce')
    wanlian_df_test['listed_date'] = pd.to_datetime(wanlian_df_test['listed_date'], errors='coerce')
    wanlian_df_test = wanlian_df_test.dropna(subset=['five_class_result_adj'])
    wanlian_df_test = wanlian_df_test.dropna(subset=['rolling_avg_float_market_value_rank'])
    
    logger.info('开始万联规则判断，生成万联分布')
    if 'discount_rate' in wanlian_df_test.columns:
        wanlian_df_test["wanlian_rating"] = wanlian_df_test.apply(wanlian_grade_judge_basedeloitte, axis=1)
    else:
        wanlian_df_test["wanlian_rating"] = wanlian_df_test.apply(wanlian_grade_judge_basedeloitte_early, axis=1)
    
    logger.info('万联评级生成完毕！')
    
    wanlian_df_test = wanlian_df_test[['stock_code', 'trade_date', 'five_class_result_adj', 'wanlian_rating']]
    wanlian_df_test['trade_date'] = wanlian_df_test['trade_date'].dt.strftime('%Y-%m-%d')
    insert_data_without_duplicates(wanlian_df_test, 'saas_rating')
    empty_folder('temp_model_backtesting')
    
    logger.info('万联评级保存在result文件夹rating_db.db下,程序运行完毕!')


# 万联评级判断程序

def wanlian_grade_judge(row):     #当前版本弃用！
    if row['td_mkt']=='北交所':
        return 'E'
    elif (row['listed_board_name'] in ['创业板', '科创板'] and row['st_issuance']== '注册制' 
        and row['five_class_result_adj'] == 'E'):
        return 'E'
    elif row['td_mkt']!='北交所' and row['five_class_result_adj'] == 'E':
        return 'E'
    elif pd.Timestamp(row['listed_date']) > pd.Timestamp(row['trade_date']) - pd.Timedelta(days=7):
        return 'E'
    elif ('沪深300' in row['index_name'] and
    row['listed_board_name']=='主板' and
    row['rolling_avg_float_market_value_rank']<=0.03 and
    row['rolling_avg_trans_amt_rank']<=0.1 and
    row['five_class_result_adj'] in ['A', 'B'] and
    row['discount_rate'] >= 0.6 and
    row['goodwill_ratio']<0.3 and
    row['collateral_ratio']<20):
        return 'A'
    elif (((row['index_name_1']=='中证800' and row['listed_board_name']=='主板') or
         (row['st_issuance'] == '核准制' and row['listed_board_name'] == '创业板')) and
         row['rolling_avg_float_market_value_rank']<=0.2 and
         row['rolling_avg_trans_amt_rank']<=0.2 and
         row['five_class_result_adj'] in ['A', 'B','C'] and
         row['discount_rate'] >= 0.5 and
         row['goodwill_ratio']<0.3 and
         row['collateral_ratio']<30):
         return 'B'
    elif (row['td_mkt'] in ['上交所', '深交所'] and
          row['st_issuance'] != '注册制' and
          row['rolling_avg_float_market_value_rank'] <= 0.80 and  # 日均流通市值排名前80%
          row['rolling_avg_trans_amt_rank'] <= 0.90 and  # 日均成交量排名前90%
          row['five_class_result_adj'] in ['A', 'B', 'C']) :
        return 'C'
    elif (row['td_mkt'] in ['上交所', '深交所'] and
          row['st_issuance'] == '注册制' and
          row['listed_date'] <= pd.Timestamp(row['trade_date']) - pd.Timedelta(days=90) and  # 上市满3个月
          row['rolling_avg_float_market_value_rank'] <= 0.50 and  # 日均流通市值排名前50%
          row['rolling_avg_trans_amt_rank'] <= 0.50 and  # 日均成交量排名前50%
          row['five_class_result_adj'] in ['A', 'B', 'C'] ):
        return 'C'
    elif (row['listed_board_name'] == '主板' and
          row['st_issuance'] == '注册制' and
          pd.Timestamp(row['trade_date']) - pd.Timedelta(days=6) <= pd.Timestamp(row['listed_date']) <= pd.Timestamp(row['trade_date']) - pd.Timedelta(days=90)):
        return 'C'
    else:
        return 'D'
    
def wanlian_grade_judge_basedeloitte(row):     #使用该判断函数
        return 'D'
    
def wanlian_grade_judge_basedeloitte_early(row):      # 处理更早无同业数据的判断函数
        return 'D'

# 主程序运行
#####################初始化查询配置,此处修改时间区间#####################
def daily_query_update():
    query_config=init(start_date=None, end_date=None)
    end_date = pd.read_sql("SELECT MAX(td) FROM model_st ms ;",query_config['istock_conn'])
    end_date=end_date.loc[0][0].strftime('%Y-%m-%d')
    new_date=pd.read_sql('SELECT MAX(trade_date) FROM saas_rating;',query_config['rating_conn'])
    new_date=new_date.loc[0][0]
    if pd.to_datetime(new_date)==pd.to_datetime(end_date):
        logger.info('no new data')  
    else: 
        start_date=pd.read_sql(f'''SELECT MIN(td)  FROM model_st
                               WHERE td > '{new_date}'; ''',query_config['istock_conn'])
        start_date=start_date.loc[0][0].strftime('%Y-%m-%d')
        query_config=init(start_date=start_date, end_date=end_date)
        get_saas_data(query_config)
#######################################################################


####################每日更新数据时启用#######################
daily_query_update()


