from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

def drow_circle():
    with st.echo(code_location='below'):
        total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
        num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

        Point = namedtuple('Point', 'x y')
        data = []

        points_per_turn = total_points / num_turns

        for curr_point_num in range(total_points):
            curr_turn, i = divmod(curr_point_num, points_per_turn)
            angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
            radius = curr_point_num / total_points
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            data.append(Point(x, y))

        st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
            .mark_circle(color='#0068c9', opacity=0.5)
            .encode(x='x:Q', y='y:Q'))
 
def test_write(): #write功能展示
    '''
    st.write()
    st.write()是一个泛型函数，根据传入对象不同采取不同的展示方式，比如传入pandas.DataFrame时，st.write(df)默认调用st.dataframe()，传入markdown时，st.write(markdown)默认调用st.markdown()；跟R的泛型函数非常类似。可传入的对象有:

    write(data_frame) : Displays the DataFrame as a table.
    write(func) : Displays information about a function.
    write(module) : Displays information about the module.
    write(dict) : Displays dict in an interactive widget.
    write(obj) : The default is to print str(obj).
    write(mpl_fig) : Displays a Matplotlib figure.
    write(altair) : Displays an Altair chart.
    write(keras) : Displays a Keras model.
    write(graphviz) : Displays a Graphviz graph.
    write(plotly_fig) : Displays a Plotly figure.
    write(bokeh_fig) : Displays a Bokeh figure.
    write(sympy_expr) : Prints SymPy expression using LaTeX.
    write(markdown):
    '''

    # 字典
    st.write({"a": [1, 2, 3],
              "b": [2, 3, 4]})

    # pandas数据框
    st.write(pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [4, 5, 6, 7, 8]
    }))

    # Markdown文字
    st.write("Hello, *World!* :sunglasses:")

    # 绘图对象
    df = pd.DataFrame(
        np.random.randn(200, 3),
        columns=["a", "b", "c"]
    )

    c = alt.Chart(df).mark_circle().encode(
        x="a", y="b", size="c", color="c", tooltip=["a", "b", "c"])
    st.write(c)

def test_table(): #动态表格展示
    # 动态表格（表格太大时只展示一部分，可滑动表格下方滑动条查看不同部分）
    # st.write默认调用st.dataframe()
    df = pd.DataFrame(
        np.random.randn(10, 20),
        columns=('col %d' % i for i in range(20)))

    st.write(df)

    # 静态表格（展示表格全部内容，太大时滑动App界面下方的滑动条查看不同部分）
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [4, 5, 6, 7, 8]
    })

    st.table(df)

    #pandas.DataFrame的style也可正常显示
    df = pd.DataFrame(
        np.random.randn(10, 20),
        columns=('col %d' % i for i in range(20)))

    st.dataframe(df.style.highlight_max(axis=0))

    #Code
    #仅展示Code，Code不执行
    code = """
    def sum_(x):
        return np.sum(x)
    """
    st.code(code, language="python")

    code = """
    for (i i 1:10) {
        print(i)
    }
    """
    st.code(code, language="r")

    st.markdown("""
    ​```python
    print("hello")
    """)
    #展示Code，同时执行Code；需要将code放入st.echo()内
    with st.echo():
        for i in range(5):
            st.write("hello")

    #使用缓存，第一次加载后，下次调用，直接从缓存中提取数据
    @st.cache
    def load_metadata():
        DATA_URL = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/labels.csv.gz"
        #DATA_URL = "labels.csv.gz"
        return pd.read_csv(DATA_URL, nrows=1000)

    @st.cache
    def create_summary(metadata, summary_type):
        one_hot_encoded = pd.get_dummies(metadata[["frame", "label"]],
                            columns=["label"])
        return getattr(one_hot_encoded.groupby(["frame"]), summary_type)()

    # Piping one st.cache function into another forms a computation DAG.
    summary_type = st.selectbox("Type of summary:", ["sum", "any"])
    metadata = load_metadata()
    #metadata.to_excel('labels.xlsx',index=False)
    summary = create_summary(metadata, summary_type)
    st.write('## Metadata', metadata, '## Summary', summary)

    #远程提取数据，并缓存，使用select来选择要显示的数据内容。
    # Reuse this data across runs!
    read_and_cache_csv = st.cache(pd.read_csv)

    BUCKET = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/"
    #BUCKET = ""
    data = read_and_cache_csv(BUCKET + "labels.csv.gz", nrows=1000)
    desired_label = st.selectbox('Filter to:', ['car', 'truck'])
    st.write(data[data.label == desired_label])

    #从S3下载数据，并根据选择，展示数据
    DATE_COLUMN = 'date/time'
    DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
                'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

    #DATA_URL = 'uber-raw-data-sep14.csv'

    @st.cache
    def load_data(nrows):
        data = pd.read_csv(DATA_URL, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
        return data

    data_load_state = st.text('Loading data...')
    data = load_data(1000)
    #data.to_csv('uber-raw-data-sep14.csv',index=False)

    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)

    st.subheader('Number of pickups by hour')
    hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
    st.bar_chart(hist_values)

    #展示数据
    option = st.selectbox(
        'Which number do you like best?',
          [1,2,3,4,5])
    # Some number in the range 0-23
    hour_to_filter = st.slider('hour', 0, 23, 17)
    filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

    st.subheader('Map of all pickups at %s:00' % hour_to_filter)
    st.map(filtered_data)



    dataframe = pd.DataFrame(
        np.random.randn(10, 20),
        columns=('col %d' % i for i in range(20)))

    '''
    dataframe 显示方式一：sr.write
    '''
    st.write(dataframe)

    '''
    dataframe 显示方式二：直接键入最终结果dataframe
    '''
    dataframe

    '''
    dataframe 显示方式三：st.dataframe
    '''
    st.dataframe(dataframe.style.highlight_max(axis=0))

    '''
    dataframe 显示方式四：st.table
    最丑的一种方式，会变成无页面约束的宽表
    '''
    dataframe = pd.DataFrame(
        np.random.randn(10, 20),
        columns=('col %d' % i for i in range(20)))
    st.table(dataframe)

def test_control(): #控件功能展示
    #各种控件

    # 数字输入框
    number = st.number_input("Insert a number", 123)

    # 单行文本输入框
    word = st.text_input("Insert a word", "123")
    st.write("The number is", number, "The word is", word)

    # 多行文本输入框
    st.text_area("Text to analyze", "I love China")

    # 日期输入框
    st.date_input("Insert a date")

    # 时间输入框
    st.time_input("Insert a time")

    # 向表单插入元素
    with st.form("my_form1"):
        st.write("我在 1 框框里~")
        slider_val = st.slider("框框滑块")
        checkbox_val = st.checkbox("pick me")
        # Every form must have a submit button.
        submitted = st.form_submit_button("1-Submit")

    # 乱序插入元素
    form = st.form("my_form2")
    form.slider("我在 2 框框里~")
    st.slider("我在外面")
    # Now add a submit button to the form:
    form.form_submit_button("2-Submit")


    # 点击按钮
    number = st.button("click it")
    st.write("返回值:", number)

    # 滑动条
    x = st.slider("Square", min_value=0, max_value=80)
    st.write(x, "squared is", np.power(x, 2))

    #或
    x = st.slider('x')
    st.write(x, 'squared is', x * x)

    """ ### 4.6 拉选框


    包括：
    - 常规滑块 - range slider
    - 时间滑块 - time slider
    - 日期选项 - datetime slider

    """


    age = st.slider('How old are you?', 0, 130, 25)
    st.write("I'm ", age, 'years old')

    # 常规滑块 - range slider
    values = st.slider(
        'Select a range of values',
        0.0, 100.0, (25.0, 75.0))
    st.write('Values:', values)


    # 时间滑块 - time slider
    from datetime import time  #此处的time功能是datetime中time模块特有
    appointment = st.slider(
         "Schedule your appointment:",
         value=(time(11, 30), time(12, 45)))
    st.write("You're scheduled for:", appointment)

    # 日期选项 - datetime slider
    start_time = st.slider(
         "请选择开始日期：",
         value=datetime.datetime(2019, 1, 1),
         format="YYYY-MM-DD")
    st.write("开始日期:", start_time)

    # 日期选项 - datetime slider
    start_time = st.slider(
         "请选择结束日期：",
         value=datetime.datetime(2021, 12, 1),
         format="YYYY-MM-DD")
    st.write("结束日期:", start_time)


    # 常规
    color = st.select_slider(
         'Select a color of the rainbow',
         options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'])
    st.write('My favorite color is', color)

    # range select slider 区域范围的选择滑块
    start_color, end_color = st.select_slider(
        'Select a range of color wavelength',
         options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'],
         value=('red', 'blue'))
    st.write('You selected wavelengths between', start_color, 'and', end_color)


    # 单文件载入
    uploaded_file = st.file_uploader("单文件载入上传")
    if uploaded_file is not None:
         # To read file as bytes:
         bytes_data = uploaded_file.read()
         st.write(bytes_data)

         # To convert to a string based IO:
         stringio = StringIO(uploaded_file.decode("utf-8"))
         st.write(stringio)

         # To read file as string:
         string_data = stringio.read()
         st.write(string_data)

         # Can be used wherever a "file-like" object is accepted:
         st.write(uploaded_file)
         dataframe = pd.read_csv(uploaded_file)
         st.write(dataframe)

    # 多文件载入
    uploaded_files = st.file_uploader("多个文件载入上传", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name)
        st.write(bytes_data)


    #颜色选择
    color = st.color_picker('颜色选择：', '#00f900')
    st.write('The current color is', color)


    '''
    ## 5 控制组件 - Control flow

    '''


    """ ### 5.1 输入框
        只有输入了，才会继续进行下去...
        """

    name = st.text_input('Name')
    if not name:
        st.warning('Please input a name.')
        st.stop()
    st.success(f'Thank you for inputting a name. {name}')

    #多个选择框 - 选上了就会上记录
    options = st.multiselect(
        'What are your favorite colors',
        ['Green', 'Yellow', 'Red', 'Blue'],
        ['Yellow', 'Red'])
    st.write('You selected:', options)

    # 检查框
    res = st.checkbox("I agree")
    st.write(res)

    # 单选框
    st.selectbox("Which would you like", [1, 2, 3])

    # 单选按钮
    st.radio("Which would you like", [1, 2, 3])

    # 多选框
    selector = st.multiselect("Which would you like", [1, 2, 3])
    st.write(selector)

    ## 气球效果
    #st.balloons()

    #上传csv文件
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)

    #下载：目前Streamlit还没有专门的下载控件，下载pandas Dataframe为csv文件可通过以下方式实现
    #点击Download CSV File便可下载文件
    data = [(1, 2, 3)]
    df = pd.DataFrame(data, columns=["Col1", "Col2", "Col3"])
    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">下载 CSV 文件</a> (点击鼠标右键，另存为 &lt;XXX&gt;.csv)'
    st.markdown(href, unsafe_allow_html=True)

    #侧边栏控件
    #以上控件大部分都有对应的侧边栏形式，如上述st.selectbox若想放置在侧边栏，可使用st.sidebar.selectbox
    # 单选框
    selector1=st.sidebar.selectbox("请选择你喜欢的选项：", [1, 2, 3], key="1")

    # 单选按钮
    selector2=st.sidebar.radio("请选择你喜欢的选项：", [1, 2, 3], key="1")

    # 多选框
    selector = st.sidebar.multiselect("请选择你喜欢的选项：", [1, 2, 3], key="3")
    st.write("你的选项分别是：\n",selector1,selector2,selector)

def test_pandas_get(): #测试pandas爬虫

    #import csv
    chart_list = ['抓取世界大学综合排名','抓取新浪财经基金重仓股数据','抓取证监会披露的IPO数据']
    sidebar = st.sidebar.selectbox(
    "请选择：",
    chart_list
    )

    if sidebar == "抓取世界大学综合排名":
        #抓取世界大学排名（1页数据）
        url1 = 'http://www.compassedu.hk/qs'
        df1 = pd.read_html(url1)[0]  #0表示网页中的第一个Table
        df1.to_csv('世界大学综合排名.csv',index=0)
        st.markdown("## 抓取世界大学综合排名")
        st.dataframe(df1)
    elif sidebar == "抓取新浪财经基金重仓股数据":
        #抓取新浪财经基金重仓股数据（6页数据）
        df2 = pd.DataFrame()
        for i in range(6):
            url2 = 'http://vip.stock.finance.sina.com.cn/q/go.php/vComStockHold/kind/jjzc/index.phtml?p={page}'.format(page=i+1)
            df2 = pd.concat([df2,pd.read_html(url2)[0]])
            print('第{page}页抓取完成'.format(page = i + 1))
        #df2.to_csv('./新浪财经数据.csv',encoding='utf-8',index=0)
        st.markdown("## 抓取新浪财经基金重仓股数据")
        st.dataframe(df2)
    elif sidebar == "抓取证监会披露的IPO数据":
        #抓取证监会披露的IPO数据（217页数据）
        from pandas import DataFrame
        import time
        st.markdown("## 抓取证监会披露的IPO数据")
        start = time.time() #计时
        df3 = DataFrame(data=None,columns=['公司名称','披露日期','上市地和板块','披露类型','查看PDF资料']) #添加列名
        for i in range(1,218):
            url3 ='http://eid.csrc.gov.cn/ipo/infoDisplay.action?pageNo=%s&temp=&temp1=&blockType=byTime'%str(i)
            df3_1 = pd.read_html(url3,encoding='utf-8')[2]  #必须加utf-8，否则乱码
            df3_2 = df3_1.iloc[1:len(df3_1)-1,0:-1]  #过滤掉最后一行和最后一列（NaN列）
            df3_2.columns=['公司名称','披露日期','上市地和板块','披露类型','查看PDF资料'] #新的df添加列名
            df3 = pd.concat([df3,df3_2])  #数据合并
            st.write('第{page}页抓取完成'.format(page=i))
        #df3.to_csv('./上市公司IPO信息.csv', encoding='utf-8',index=0) #保存数据到csv文件
        end = time.time()
        st.write('共抓取',len(df3),'家公司,' + '用时',round((end-start)/60,2),'分钟')
        st.dataframe(df3)
    else:
        st.title("Pandas抓取数据")
        st.write("欢迎使用功能强大的Pandas")
        
def main():  #主程序模块，调用其他程序模块
    st.sidebar.markdown("# 主程序导航图")

    #
    options = ['基本功能展示','write功能展示','动态表格展示','控件功能展示','绘图、图片功能展示',\
                '音频、视频功能展示','网页布局','图像目标检测demo网页','单独页面展示','数据转换',\
                '动态图形展示','跳转新页面','展示HTML文件内容','单页显示数据','项目管理','文件查找',\
                '修改配置信息','文件下载','图像识别','altair可视化数据分析','对时间序列数据集进行可视化过滤']
    object_type = st.sidebar.selectbox("请选择程序模块", options, 1)
    # min_elts, max_elts = st.sidebar.slider("多少 %s     (选择一个范围)?" % object_type, 0, 25, [10, 20])
    # selected_frame_index = st.sidebar.slider("选择一帧 (帧的索引)", 0, len(object_type) - 1, 0)
    st.sidebar.markdown("------")
    st.sidebar.markdown("## 子模块  ")
    st.sidebar.markdown("---")


    if object_type == '动态表格展示':
        
        test_table()
    elif object_type == '基本功能展示':
        drow_circle()
    elif object_type == 'write功能展示':
        test_write()
    elif object_type == '控件功能展示':
        test_control()
    elif object_type == '绘图、图片功能展示':
        test_control()
    elif object_type == '音频、视频功能展示':
        test_control()
    elif object_type == '网页布局':
        test_control()
    elif object_type == '图像目标检测demo网页':
        test_control()
    elif object_type == '单独页面展示':
        test_control()
    elif object_type == '数据转换':
        test_control()
    elif object_type == '动态图形展示':
        test_control()
    elif object_type == '跳转新页面':
        test_control()
    elif object_type == '展示HTML文件内容':
        test_control()
    elif object_type == '单页显示数据':
        test_control()
    elif object_type == '项目管理':
        test_control()
    elif object_type == 'pandas爬虫':
        test_pandas_get() #测试pandas爬虫
    elif object_type == '对时间序列数据集进行可视化过滤':
        test_control() #对时间序列数据集进行可视化过滤
        
if __name__ == '__main__':
    #drow_circle()
    main()
