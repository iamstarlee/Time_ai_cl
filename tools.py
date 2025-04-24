import pandas as pd

def count_type():
    file_path = 'data/half_case.xlsx'
    # 读取Excel表格的所有内容
    df = pd.read_excel(file_path)

    # 假设L列是 'L' 列，如果是其他列名，请修改为实际列名
    # 计算每种数据类型的数量
    type_counts = df['故障类型'].value_counts()

    # 输出每种数据类型的数量
    print(type_counts)


def split_data():
    # 读取Excel文件
    file_path = 'data/case.xlsx'
    df = pd.read_excel(file_path)

    # 确保数据每 10 行为一个数据块
    df['block_id'] = df.index // 10

    # 获取每个数据块的故障类型（仅取每个数据块的第一行）
    type_column = '故障类型'
    block_types = df.groupby('block_id').first()[type_column]

    # 按故障类型筛选数据
    A_blocks = block_types[block_types == '无'].sample(frac=0, random_state=42).index
    B_blocks = block_types[block_types == '局部放电'].sample(frac=0, random_state=42).index
    C_blocks = block_types[block_types == '高温过热'].sample(frac=0, random_state=42).index
    D_blocks = block_types[block_types == '过热缺陷'].sample(frac=0, random_state=42).index

    # E、F 需要完整保留
    E_blocks = block_types[block_types == '悬浮放电'].index
    F_blocks = block_types[block_types == '受潮故障'].index

    # 选择数据块对应的完整 10 行数据
    selected_blocks = list(A_blocks) + list(B_blocks) + list(C_blocks) + list(D_blocks) + list(E_blocks) + list(F_blocks)
    final_data = df[df['block_id'].isin(selected_blocks)]

    # 移除 `block_id` 列
    final_data = final_data.drop(columns=['block_id'])

    # 保存到新的 Excel 文件
    final_data.to_excel('data/zero_zero_case.xlsx', index=False)
    print("数据已成功提取并保存")



def plot_loss():
    import matplotlib.pyplot as plt
    import os

    # 读取保存的损失数据
    with open('logs/epoch_test_loss.txt', 'r') as f:
        lines = f.readlines()

    # 将损失数据转换为浮点数
    losses = [float(line.strip()) for line in lines]

    # 绘制损失曲线
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join("epoch_test_loss.png"))

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
    count_type()