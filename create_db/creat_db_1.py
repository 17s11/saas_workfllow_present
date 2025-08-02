import sqlite3

def create_tables_1():
    # 连接到 SQLite 数据库（如果数据库不存在，则会自动创建）
    conn = sqlite3.connect(r'result/rating_db.db')  
    cur = conn.cursor()

    # 创建 saas_indicator 表
    cur.execute('''
    CREATE TABLE IF NOT EXISTS saas_indicator (
        stock_code TEXT NOT NULL,  
        trade_date TEXT NOT NULL,  
        rolling_avg_trans_amt REAL,  
        sec_short_name TEXT,  
        rolling_avg_float_market_value REAL,  
        collateral_ratio REAL,  
        five_class_result_adj TEXT,  
        discount_rate REAL,  
        goodwill_ratio REAL,  
        listed_board_name TEXT,  
        listed_date TEXT,  
        td_mkt TEXT,  
        st_issuance TEXT,  
        index_name TEXT,  
        index_name_1 TEXT,  
        rolling_avg_trans_amt_rank REAL,  
        rolling_avg_float_market_value_rank REAL,  
        PRIMARY KEY (stock_code, trade_date)  
    );
    ''')

    # 创建 saas_rating 表
    cur.execute('''
    CREATE TABLE IF NOT EXISTS saas_rating (
        stock_code TEXT NOT NULL,  
        trade_date TEXT NOT NULL,  
        five_class_result_adj TEXT,  
        wanlian_rating TEXT,  
        PRIMARY KEY (stock_code, trade_date)  
    );
    ''')

    # 提交事务，确保所有更改都被保存到数据库
    conn.commit()

    # 关闭与数据库的连接
    conn.close()
