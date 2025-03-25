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

def plot_loss():
    import matplotlib.pyplot as plt
    import os

    # 读取保存的损失数据
    with open('logs/output_file_100.txt', 'r') as f:
        lines = f.readlines()

    # 将损失数据转换为浮点数
    losses = [float(line.strip()) for line in lines]

    # 绘制损失曲线
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join("epoch_loss_100.png"))

def extract_last_coloumn():
    # 打开txt文件并读取内容
    with open('logs/epoch_loss_100.txt', 'r') as file:
        lines = file.readlines()

    # 提取每行的最后一列
    last_column = []
    for line in lines:
        # 去掉行末的换行符，并按空格或制表符分隔列
        columns = line.strip().split()  # 如果列是由空格或制表符分隔
        if columns:
            last_column.append(columns[-1])  # 获取每行的最后一列

    # 将最后一列保存到新的txt文件
    with open('logs/output_file_100.txt', 'w') as output_file:
        for item in last_column:
            output_file.write(item + '\n')  # 每个元素写入新文件，并换行



if __name__ == '__main__':
    plot_loss()
    