#######################从github下载下来,务必先运行该文件初始化!#####################
from dateutil.relativedelta import relativedelta
from datetime import datetime

from create_db import create_tables
from module_query_update import getConnection
from data_logger import setup_logger
from module_query_update import daily_query_initialize
from module_create_discountrate import caluculate_discount_rate_initialize

import warnings
warnings.filterwarnings('ignore')
###############################参数修改################################
init_config = {'data_range':3,  ##修改数据X，回溯前x月的数据进入数据库，建议3个月及以上,5个月初始化需要25min,建议5个月
               'is_report':1,          ##是否生成上一期报告(样本报告)，1为生成，0为不生成
               'clean_log':0          ##是否清空日志(保留现有的data_log)，1为清除，0为不清除,推荐清除
               }
########################################################################
def main(init_config):
    current_date = datetime.now().date()
    range_date = current_date - relativedelta(months=init_config['data_range'])

    if init_config['clean_log'] == 1:      # 清空日志判断
        with open('log\data_log.log', 'w') as file:
            file.close()
        logger.info('data_log.log文件已清空! 此处以下是最新的日志!')
    else:
        pass

    create_tables()             ## 创建数据库rating_db.db
    logger.info('rating_db.db创建完成! 文件路径为: result/rating_db.db')

    # rating_eninge = getConnection('rating_conn')
    # istock_engine =  getConnection('istock_conn') 

    daily_query_initialize(range_date)      ##五级分类数据初始化,入库初始化
    logger.info('五级分类数据配置初始化完成!')

    caluculate_discount_rate_initialize(range_date)    ##折算率入库初始化
    logger.info('折算率入库配置初始化完成!')

    logger.info('SaaS取数程序_运行版配置初始化完成!')

#########################################################################
if __name__ == '__main__':
    logger = setup_logger()
    main(init_config)
