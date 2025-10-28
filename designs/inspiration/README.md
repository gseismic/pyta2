# 定位
可简单应用于机器学习的指标库

# 设计
* 包含输出参数个数、最大值、最小值、对称性
* 输入变动、输出的快速估算
* 与Roll 相反的模型是Stream/Moving

```
Deep Reinforcement Learning for Automated Stock Trading
https://towardsdatascience.com/deep-reinforcement-learning-for-automated-stock-trading-f1dad0126a02

https://www.coursera.org/learn/trading-strategies-reinforcement-learning/lecture/xYDtB/introduction-to-course

全参数化优化 (网络设计-TradeFlow）
(多尺度)
 close			   low	
	\			 	|
	 \				|			
	  \				|			
	   \			|		
		SMA(m)	  SMA(m)	
		  \			/
		  	\	  /
			Ixx(sma_close, sma_low)
			   |
			   |
			   .
			   .
			   .
		买入/卖出/持有


  (t, Price)
  	|
	|
	v
(可以自带特征评价)
特征网络(提取数据特征；F: Stream -> Feature; 原始数据 --> 特征数值 numeric)
	|
	|
	v
归一化网络(归一化特征数值(可以多个特征组合);  N: Feature -> LogicFeature; -->bool[0~1])
	|
	|
	v
决策网络(计算归一化特征下各个动作的价值函数Q; Q: LogicFeature -> ActionValue;  bool --> Action 简化，if/else; bools -> bools)
	|
	|
	v
评价网络(不同时间尺度下，动作的累计价值函数, R: (Env(Price, Balance, Fee), action=policy_choice{epsilon}(ActionValue)) -> Reward (真实预期收益)
step reward
(Env包括：价格Price，账户状况Balance, 手续费Fee，评价函数怎么定？)
 （不同DNA/风险偏好性格/时间尺度(长线、短线交易者): 智能体的状态、做各种动作可能引起的状态改变）
	|
	|
	v
 累计收益r
 (Argmax_{actions}(Reward))
 r = R((Price, Balance), action=choice(ActionValue = Q( LogicFeature = N(Feature = F(Stream) ) )))


 (t, Price) 发生变化，应该分别采取什么动作才能使不同时间尺度的Agent1/Agent2/... 收益最大化
 —— t,Price如果能够预测，特定DNA的Agent的策略就是固定了

本质：市场中Price变化后

(老SIM卡含有文档数据，思考文档)
评价函数:
	给出`涨跌概率图`
	已知`涨跌概率图`，给出策略


广义神经网络
Max(SMA(m)(close), SMA(m)(low))

固定结构网络--优化网络参数
(不同)


关系型神经网络
	可以处理if/else问题

价值函数
按照双均线一定盈利吗，什么情况下一定盈利（价格分解？）

def on_init(self):
	self.highs = Roll(10)
	self.lows= Roll(10)
	self.I = {
		'sma_h': rSMA(10, self.highs)
		'ema_l': rEMA(10, self.lows)
	}

def compile(self):
	pass

def on_data(self, high, low):
	self.highs.push(high)
	self.lows.push(low)

	self.I
```
v = rSMA(10, Roll(10)(close))

r = Roll(3)
r.push(v)


## 文件夹
pyta2
	- classic
	- r2021
	- volume

## 样例
```python

class rSMA:

	def __init__(self, params):
		self.params = params
	
	@property
	def required_input(self):

	def rolling(self, values):
		...
		output = {'value': value}
		return output
	
	@property
	def meta(self):
		return {
			'value': {
				'max': None,
				'min': None,
				'mean': None
			}
		}
```
