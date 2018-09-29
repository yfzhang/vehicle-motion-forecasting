## branch说明
### v0
- 用FCN网络,有spatial learning
- 输入过去的trajectory和全部点云地图上的特征(height,var,red,green,blue)

### v1
- 根据过去的trajectory提取方向和速度信息.方向信息又sin和cos来表示.
- 把速度,sin,cos信息分别放成一个layer提供给network.(速度layer的每个grid都是同一个速度值)

### v2
- 根据过去的trajectory(方向,速度),预测一个简单的轨迹,直接当做一层feature提供给网络

### v3
- 按照activity forecasting里的做法,每次value iteration的时候,把goal state的value重置为0

### v4
- 把速度,方向,和位置相关的信息,直接输入到最后几层做regression的layer
- 前面几层卷积层用来提取和geometry,RGB相关的feature

### v5
- 重新整理训练数据,平衡拐弯和直线的比例

### v6
- 添加新的训练数据 (平衡在岔路口的选择,在宽阔路段的S形drive)
  - 6.1, 加pre-train
    - 6.11, 正常的Q; discount 0.95
    - 6.12, kinematic信息, x,y从0到1, 没负数
    - 6.13, kinematic信息, x,y从-2到2; Q乘10; discount 0.95
    - 6.14, kinematic信息, x,y从-1到1; Q乘20; discount 0.95
  - 6.2, 增加narrow trail的数据
    - 6.23, (修正)kinematic信息, x,y从-1到1, feat_out=25, regression_hidden_size=64
    - 6.25, 删去training里面的S型
    - 6.26, 重新train
    - 6.27, 用tangent
  - 6.3, random加uniform环境,只有kinematics, random=0.5, tanget
    - 6.31, random=0.2, tangent
    - 6.32, random=0.2, no tagent
    - 6.33, random=0.3, 删S型(修正)
