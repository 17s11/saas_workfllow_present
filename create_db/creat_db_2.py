import sqlite3

def create_tables_2():
    # 连接到 SQLite 数据库（如果数据库不存在，则会自动创建）
    conn = sqlite3.connect(r'result/rating_db.db')  
    cur = conn.cursor()

    # 创建 broker_tag_sheet 表
    cur.execute('''
    CREATE TABLE IF NOT EXISTS broker_tag_sheet (
        stock_code TEXT NOT NULL,
        stock_name TEXT NOT NULL,
        trade_date TEXT NOT NULL,
        SAAS_rating TEXT,
        中信建投 TEXT,
        华泰证券 TEXT,
        国泰君安 TEXT,
        tag INTEGER,
        PRIMARY KEY (stock_code, trade_date)
    );
    ''')

    # 提交事务，确保所有更改都被保存到数据库
    conn.commit()

    # 关闭与数据库的连接
    conn.close()

# if __name__ == "__main__":
#     create_tables()