import sqlite3

def create_tables_3():
    # 连接到 SQLite 数据库（如果数据库不存在，则会自动创建）
    conn = sqlite3.connect(r'result/rating_db.db')  
    cur = conn.cursor()

    # 创建 broker_tag_sheet 表
    cur.execute('''
    CREATE TABLE edit_log (
        stock_code TEXT,
        stock_name TEXT,
        trade_date TEXT,
        SAAS_rating TEXT,
        中信建投 TEXT,
        华泰证券 TEXT,
        国泰君安 TEXT,
        tag TEXT,
        AI_comment TEXT,
        AI_rating TEXT,
        edit_rating TEXT,
        edit_date TEXT,
        PRIMARY KEY (stock_code, trade_date)
    );
    ''')

    cur.execute('''CREATE TABLE new_discount_record (
    id INTEGER, -- 移除了AUTOINCREMENT和PRIMARY KEY，因为现在使用复合主键
    trade_date TEXT NOT NULL, -- 日期列，格式为'yyyy-mm-dd'
    stock_code TEXT NOT NULL,
    default_probability REAL NOT NULL,
    conversion_rate REAL NOT NULL,
    conversion_rate_adj REAL NOT NULL,
    rule_001 REAL NOT NULL,
    rule_002 REAL, -- 允许NULL值
    rule_003 REAL, -- 允许NULL值
    rule_004 REAL, -- 允许NULL值
    rule_005 REAL, -- 允许NULL值
    rule_006 REAL, -- 部分非空
    rule_007 REAL, -- 允许NULL值
    rule_008 REAL, -- 部分非空
    base_line_1 REAL NOT NULL,
    rating_weight REAL, -- 部分非空
    base_line_2 REAL, -- 部分非空
    fin_conversion_rate REAL NOT NULL,
    five_class_result_saas_adj TEXT, -- 部分非空
    PRIMARY KEY (trade_date, stock_code) -- 定义复合主键
    );
    ''')

    # 提交事务，确保所有更改都被保存到数据库
    conn.commit()

    # 关闭与数据库的连接
    conn.close()

# if __name__ == "__main__":
#     create_tables()