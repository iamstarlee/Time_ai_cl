# 设备运维连续学习算法
本算法基于数据重放（Replay-based Approach）实现了模型的连续学习化，基础类为“无”、“局部放电”、“高温过热”和“过热缺陷”，需要连续学习的类为“悬浮放电”和“受潮故障”。数据重放率为超参数，根据经验可以设为100%、75%、50%、25%和0%。下面介绍重放率为50%时的使用方法。

## 训练和测试基础类
1. 首先注释掉`Dataload_images.py`中209行至224行“受潮故障”和“悬浮放电”两个分支，仅加载基础类数据。
2. 然后修改`main_with_img.py`中的`train()`方法中需要加载的数据xlsx文件，task设置为type，运行`pyhon main_with_img.py --task type`。

## 训练和测试连续学习类
1. 首先运行`tools.py`中的`split_data()`方法划分新数据。
2. 然后task设置为type_cl。
3. 取消注释ataload_images.py中209行至224行“受潮故障”和“悬浮放电”两个分支，因为此时需要连续化这两类。
4. 运行`pyhon main_with_img.py --task type_cl`。
