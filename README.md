# saas_workfllow_present
报告生成发送present版  --   删除大部分代码，仅留下部分sample
### 文件解释
- create_db 文件夹含有创建数据库的.py文件,了解数据结构可看，不可运行；

- log文件夹是存放日志的，可查看运行日志；

- password_attention中存有代码运行的各种固定数据，如果修改，可以编辑；
    - 其中attention_list.xlsx文件是同业手动对比的 舆情高关注列表，在盯市时，将负面丑闻的票放入其中(e.g. 10/03/2025的600881.SZ)
    - 切记股票代码是同花顺代码！

- report 文件夹存储着运行的同业对比报告和手动调整意见，请确保其中含有本周你运行的同业对比报告，如没有，前去broker_tag_compare.py生成

- result 文件夹是rating_db.db所在路径，可查看rating_db.db的数据库内容。

- temp_model_backtesting 文件夹是存储临时运行文件的路径。

- 单元测试五级分类.ipynb 是测试文件，用于测试机跑saas_rating和领鸟跑出的model_st数据的对比；(10-20只不一致很正常，是分位点计算的微小差异，内有归因程序cell)

- 手改覆盖修改程序.ipynb 和 broker_tag_compare.py 一般是连在一起使用；broker_tag_compare.py 运行出初步报告进入report文件夹。
- 运行手改覆盖修改程序.ipynb 可进行进一步调整，并生成ai prompt,完成手动评级调整后；运行最后单元格完成saas_five_class_white_list 表更新,最后会自动发邮件给明非姐留档！（生产环境，务必在一天内完成准确的手动更新！）

- 取数程序_saas_rating历史数据.ipynb 和 daily_query_update.py 是saas_rating取数程序，可运行；ipynb文件是取历史数据，py文件是取最新数据。(如要更新历史数据，那需要删除历史数据后入库)

- creat_new_discountrate.py 是 fin_discount_rate 取数程序，可运行；  

- data_logger.py 是日志记录程序，可运行,不可修改；

- moudules 文件夹是存放各种模块的，可运行 ；
    - module_broker_tag_compare.py 有生成初步对比报告的函数，以及获取手工调整后机跑结果的评级 的函数get_rating 
    - module_query_update.py 和 module_create_discountrate.py 中最新数据入库的函数，也有生成历史评级df的函数(不推荐使用，直接数据库取更方便)

- initialize.py是很重要的，每个文件初次下载下来是没有数据库的，需要通过这个初始化程序创建数据库，并把数据生成到其中。


### 历史版本修改记录

    1.替换白名单raw_data数据来源为istock_db       --后续：无需修改，此处取的是未经白名单调整纯机跑数据，此处使用rating_db合理！

    2.百名单数据记录入库顺序放到后面（真正白名单入库的地方）     --后续：修改broker_tag_sheet入库时间至白名单入库后，放入手改覆盖修改程序.ipynb。

    3.更换上期报告日期获取源为mysql，并且机器回溯上一期edit_df   --后续：使用sql得到上一期update日期，并回溯历史当天的调整结果，这样report文件夹中不再需要严格的文件管理

    4.清空数据库，并保证用户初始化时可以运行更新到最新日期数据        --已经清空数据库，

    6.手改覆盖程序加入新的if语句，确定上一期日期追溯正确。        --后续：已经顺利加入！加上initialize.py,在下载下文件后第一时间运行，配置必要的历史数据

    7.每次修改后，自动邮件给明非姐留档        --后续：已完成，复用周报代码

    8.五级分类单元测试.ipynb 的五级分类要换上纯机跑的rating        --后续：弃用module_broker_tag_compare.get_rating，改用five_class_result_saas_adj_ori字段，修改完成。

    9.修改end_date的判断规则，不能使用shsz_daily_qutation，而应该使用model_st，这样非交易时间后也能运行  --后续：所有的end_date取数换成了：SELECT MAX(td) FROM model_st 

    10.日志logger说明要变得更加详细                --后续：每个初始化名称加上前缀来加以区分是哪个函数运行的

### 祝愿大家哈哈气气发大财！    ---owenychen
