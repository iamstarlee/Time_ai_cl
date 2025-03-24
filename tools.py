import pandas as pd

def count_type(file_path):
    # 读取Excel表格的所有内容
    df = pd.read_excel(file_path)

    # 假设L列是 'L' 列，如果是其他列名，请修改为实际列名
    # 计算每种数据类型的数量
    type_counts = df['故障类型'].value_counts()

    # 输出每种数据类型的数量
    print(type_counts)

def half_split():
     # 读取Excel文件，假设文件名为 'data.xlsx'
    file_path = 'data/case.xlsx'

    # 读取Excel表格的所有内容
    df = pd.read_excel(file_path)

    # 假设L列是数据类型列，如果不是L列，修改为实际列名
    type_column = '故障类型'

    # 确保每条数据由10行组成，使用groupby来按每10行分组（数据从0开始计数）
    # 创建一个新的列表示数据块的编号
    df['block_id'] = df.index // 10

    # 获取每个数据块的类型
    df['block_type'] = df.groupby('block_id')[type_column].transform('first')

    # 筛选A、B、C、D的所有数据（A、B、C、D仅提取一半数据块），E和F提取所有数据块
    A_data = df[df['block_type'] == '无']
    B_data = df[df['block_type'] == '局部放电']
    C_data = df[df['block_type'] == '高温过热']
    D_data = df[df['block_type'] == '悬浮放电']
    E_data = df[df['block_type'] == '过热缺陷']
    F_data = df[df['block_type'] == '受潮故障']

    # 对A、B、C、D，只取一半的数据块
    A_data_half = A_data.groupby('block_id').head(1)  # 取每个数据块的一行数据
    B_data_half = B_data.groupby('block_id').head(1)
    C_data_half = C_data.groupby('block_id').head(1)
    D_data_half = D_data.groupby('block_id').head(1)

    # 合并A、B、C、D的半数据与E、F的所有数据
    final_data = pd.concat([A_data_half, B_data_half, C_data_half, D_data_half, E_data, F_data])

    # 按原顺序排序
    final_data = final_data.sort_index()

    # 将结果保存到一个新的Excel文件
    final_data.to_excel('data/half_case.xlsx', index=False)

    print("数据已成功提取并保存")


if __name__ == '__main__':
    half_split()
    