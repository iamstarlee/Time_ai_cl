# 设备运维连续学习算法
本算法基于数据重放（Replay-based Approach）实现了模型的连续学习化，基础类为“无”、“局部放电”、“高温过热”和“过热缺陷”，需要连续学习的类为“悬浮放电”和“受潮故障”。数据重放率为超参数，根据经验可以设为100%、75%、50%、25%和0%。下面介绍重放率为50%时的使用方法。

## 1. 训练和测试基础类
1. 首先注释掉`Dataload_images.py`中209行至224行“受潮故障”和“悬浮放电”两个分支，仅加载基础类数据。
2. 然后修改`main_with_img.py`中的`train()`方法中需要加载的数据xlsx文件，task设置为type，运行`pyhon main_with_img.py --task type`。

## 2. 训练和测试连续学习类
1. 首先运行`tools.py`中的`split_data()`方法划分新数据。
2. 然后task设置为type_cl。
3. 取消注释ataload_images.py中209行至224行“受潮故障”和“悬浮放电”两个分支，因为此时需要连续化这两类。
4. 运行`pyhon main_with_img.py --task type_cl`。

# 实验效果
### 重放率为50%时连续学习化两类的每类平均准确率
<table>
    <tr>
        <td>类别ID</td> 
        <td>故障类型</td> 
        <td>基础类准确率</td> 
        <td>连续学习准确率</td> 
   </tr>
   <tr>
        <td>1</td> 
        <td>无</td> 
        <td>94.28%</td> 
        <td>93.33%</td> 
   </tr>
   <tr>
        <td>2</td> 
        <td>局部放电</td> 
        <td>97.52%</td> 
        <td>96.88%</td> 
   </tr>
   <tr>
        <td>3</td> 
        <td>高温过热</td> 
        <td>95.53%</td> 
        <td>98.67%</td> 
   </tr>
   <tr>
        <td>4</td> 
        <td>过热缺陷</td> 
        <td>97.39%</td> 
        <td>96.92%</td> 
   </tr>
   <tr>
        <td>5</td> 
        <td>悬浮放电</td> 
        <td>-</td> 
        <td>94.12%</td> 
   </tr>
   <tr>
        <td>6</td> 
        <td>受潮故障</td> 
        <td>-</td> 
        <td>94.44%</td> 
   </tr>
    <tr>
  		 <td>平均准确率</td> 
         <td></td> 
         <td>96.18%</td> 
      	 <td>96.58%</td> 
    </tr>
    
</table>

### 连续学习化两类的总体平均准确率
<table>
    <tr>
        <td>重放率</td> 
        <td>策略</td> 
        <td>基础学习四类</td> 
        <td>连续学习两类</td> 
        <td>连续学习前后相差</td> 
   </tr>
   <tr>
        <td>100%</td> 
        <td>数据重放</td> 
        <td>96.18%</td> 
        <td>94.28%</td> 
        <td>-1.90%</td> 
   </tr>
   <tr>
        <td>75%</td> 
        <td>数据重放</td> 
        <td>96.18%</td> 
        <td>94.69%</td> 
        <td>-1.49%</td> 
   </tr>
   <tr>
        <td>50%</td> 
        <td>数据重放</td> 
        <td>96.18%</td> 
        <td>96.58%</td> 
        <td>+0.40%</td> 
   </tr>
   <tr>
        <td>25%</td> 
        <td>数据重放</td> 
        <td>96.18%</td> 
        <td>95.35%</td> 
        <td>-0.83%</td> 
   </tr>
   <tr>
        <td>0%</td> 
        <td>数据重放</td> 
        <td>96.18%</td> 
        <td>99.82%</td> 
        <td>+3.64%</td> 
   </tr>

</table>