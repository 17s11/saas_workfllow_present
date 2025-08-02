# logger.py

import logging

def setup_logger():
    logger = logging.getLogger('my_logger')
    if not logger.handlers:  # 确保handler只被添加一次
        logger.setLevel(logging.DEBUG)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        
        fh = logging.FileHandler('log\data_log.log', mode='a', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        
        logger.addHandler(ch)
        logger.addHandler(fh)
        
        logger.propagate = False  # 防止日志消息传递给父logger，避免重复记录
        
    return logger