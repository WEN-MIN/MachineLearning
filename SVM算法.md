#### 支持向量机(SVM)算法的基本思想
不管是在哪种假设前提下，SVM的分类是在一个空间中（样本空间、特征空间）找到一个超平面，将不同类别的样本划分开。与传统的回归问题处理方式不一样的是，SVR并非是基于模型输出f(x)与真实值y之间的差别来计算损失，而是在f(x)的宽度为2```$\epsilon$ ```的间隔带内，只要训练样本落入此间隔带，则认为是被预测正确了。主要的任务还是去找到分割超平面以及间隔带，以及优化的目标是最大化间隔分离超平面，让数据分类的错误率尽可能少。

函数间隔：(用来描述分类的正确性以及确信度)

给定的训练数据集T和超平面(w, b), 定义超平面(w, b)关于样本点(```$x_i, y_i$```)的函数间隔为：
```$\hat{\gamma_i} = y_i(w·x_i+b)$ ```

几何间隔：
给定的训练数据集T和超平面(w, b), 定义超平面(w, b)关于样本点(```$x_i, y_i$```)的几何距离为：
```$\gamma_i = y_i(\frac{w}{\parallel w \parallel} \cdot x_i+\frac{b}{\parallel w \parallel}) $```, 超平面(w,b)关于数据集T的几何间隔表示为超平面到所有样本点的几何间隔之最小值：
```$\gamma = \mathop{\min}\limits_{i=1,2,...,N} \gamma_i  $```

间隔最大化：我们这里假设分离超平面的方程为```w*x+b=0 ```，支持向量机的基本思想是：****求解能够正确划分训练集并且几何间隔最大的分离超平面****。
最大间隔超平面的约束问题可以表述为：

```$\max\limits_{w,b}  \gamma$```

```s.t```
``` $ y_i\frac{w}{w}·x_i+\frac{b}{w} \geq \gamma; i=1,2,...N $ ```

根据函数间隔和几何间隔的关系:

```$ \gamma = \frac{\hat{\gamma_i}}{\parallel w \parallel} $```

将该关系带入到不等式约束以及优化目标函数里面，消去不等式约束的```$\parallel w \parallel$ ```, 可以将优化目标写为：

```$\max\limits_{w,b}  \frac{\hat{\gamma}}{w}$```,

不等式约束为：

```$ y_i(w·x_i+b) \geq \hat{\gamma} $ ``` 

实际上，这里的``` $\hat{\gamma}$```对不等式约束没有影响，所以不论``` $\hat{\gamma}$```怎么改变（w,b也会相应改变），最终的不等式约束不变。这样就可以选择

``` $\hat{\gamma}=1$```

带入上面的最优化问题。 同时需要注意的是，最大化```$\frac{1}{\parallel w \parallel} $```和最小化```$\frac{1}{2} \parallel w \parallel ^2$ ```

是等价的，于是才得到以下的最优化问题形式：

```$\min\limits_{w,b} \frac{1}{2}\parallel w \parallel ^2$```

```s.t```
``` $  y_i(w·x_i+b)-1 \geq 0; i=1,2,...N $ ```
该表达式也是线性可分支持向量机的学习算法。


###### 间隔与支持向量
![1](https://note.youdao.com/yws/api/personal/file/WEBcc637465538c9ea39348c4b73272406d?method=getImage&version=2911&cstk=m06jx4Jf)
样本点在分界间隔上的点为支持向量，我们在寻找一个分割超平面的时候，需要考虑的是尽可能的让分割超平面对所有的数据有较强的容忍性。也就是尽可能的降低分类出错率。
### SVC
##### 1.线性可分支持向量机

假设样本空间线性可分，通过硬间隔最大化学习一个线性的分类器模型，即线性可分支持向量机，该模型的最优解存在且唯一，**训练集的样本点与分离平面距离最近的样本点的实例称为支持向量**


线性可分支持向量机学习算法(原问题)

```$ \min\limits_{w,b} \frac{1}{2}\parallel w \parallel ^2 $```

```s.t```
``` $  y_i(w·x_i+b)-1 \geq 0; i=1,2,...N $ ```
求解得到最优解```$w*$```,``` $b*$ ```, 得到分离超平面```w*×x+b*=0 ```

**对偶算法**：

对原问题应用拉格朗日对偶性，通过求解对偶问题得到原始问题的最优解。

> 为什么要这样做？？？
> 1. 对偶问题往往更容易求解
> 2. 方便自然引入核函数，进而推广到非线性分类问题
> 另外的解释：
> 3. 对偶问题将原始问题中的不等式约束转为了对偶问题中的等式约束；
> 4. 改变了问题的复杂度。由求特征向量w转化为求比例系数a，在原始问题下，求解的复杂度与样本的维度有关，即w的维度。在对偶问题下，只与样本数量有关。(这也体现了对于高维数据特性比较方便，但是养不数据量过大也不是好情况）

原始问题的对偶形式:

**原始问题**：
```math
\min\limits_{w,b} \frac{1}{2}\parallel w \parallel ^2
```
```$ s.t.$```

```math
y_i(w·x_i+b)-1 \geq 0; i=1,2,...N
```

构建对偶问题步骤：
1. 对每个不等式约束引入拉格朗日乘子```$a_i, i=1,2,...,N$ ```,定义拉格朗日函数：

```math
L(w,b,a) = \frac{1}{2} \parallel w \parallel ^2 - \sum_{i=1}^{N} {a_iy_i(w\cdot x_i+b)} + \sum_{i=1}^{N}a_i
```
其中```$a = (a_1, a_2,...,a_n)$```
为拉格朗日乘子。
现在对偶问题的形式为：
```math
\max_{a}\min_{w,b} L(w,b,a)
```
先求解L对w,b的极小，再求对a的极大。

2. 求解```$ \min_{w,b} L(w,b,a) $ ``` :
拉格朗日函数L分别对w,b求偏导，并令其等于0。
得到：
```math
w = \sum_{i=1}^{N} a_iy_ix_i;

\sum_{i=1}^{N} a_iy_i = 0
```
将其带入拉格朗日函数，得到：
```math
\min_{w, b} L(w,b,a)= -\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N} a_ia_jy_iy_j(x_i \cdot x_j)+\sum_{i=1}^{N} a_i
```
求```$\min_{w, b} L(w,b,a)$```对```$a$```的极大，

即**对偶问**题：
```math
\max_{a}  -\frac{1}{2} \sum_{i=1}^{N}\sum_{j=1}^{N} a_ia_jy_iy_j(x_i\cdot x_j)+\sum_{i=1}^{N} a_i
```
等价于：
```math
\min_{a}  \frac{1}{2} \sum_{i=1}^{N}\sum_{j=1}^{N} a_ia_jy_iy_j(x_i\cdot x_j)-\sum_{i=1}^{N} a_i
```


``` $s.t$ ```
```math
\sum_{i=1}^{N} a_iy_i=0

a_i \geq 0, i=1,2...N
```
先求得最优解``` $a = (a_1,...,a_N)^T$ ```
然后在计算```$w, b$ ```
向量w:
```math
w = \sum_{i=1}^{N} a_iy_ix_i
```
并选择一个```$a$ ```的正分量```$a_j > 0$ ```,计算:
```math
b=y_j-\sum_{i=1}^{N} a_iy_i(x_i \cdot x_j)
```

针对问题的不等式约束，对偶问题求解满足的**KKT条件**：
```math

a_i(y_i(w\cdot x_i +b)-1) =0 ,i=1,2,...N

y_i(w\cdot x_i +b)-1 =0 ,i=1,2,...N

a_i \geq 0  ,i=1,2,...N
```

#### 2.线性支持向量机：
假设样本空间近似线性可分，通过软间隔最大化，引入松弛变量，使其可分，即为软间隔支持向量机。线性支持向量机的解w唯一，但b不唯一.
很多实际情况下的应用并不是完全的线性可分，总会存在一些异常点，不满足函数间隔大于等于1的约束条件，为了解决这个问题，可以对每个样本点引进一个松弛变量``` $\xi_i \geq 0 $```, 使得函数间隔加上松弛变量大于等于1，这样约束条件就变成为：
```math
y_i(w \cdot x_i+b) \geq 1-\xi_i
```
对于每一个松弛变量``` $\xi_i$ ```，支付一个代价``` $\xi_i$```，目标函数变为
```math
\frac{1}{2}\parallel w \parallel ^2+C\sum_{i=1}^{N} \xi_i
```
这里的C>0是惩罚系数，C越大时，对误分类的惩罚增大，C越小时，对误分类的惩罚减小。

**该问题的原问题是**：
```math
\min_{w, b, \xi}  \frac{1}{2}\parallel w \parallel ^2+C\sum_{i=1}^{N} \xi_i
```
``` $s.t.$ ```
```math
y_i(w \cdot x_i+b) \geq 1-\xi_i,
i=1,2...N

\xi_i \geq 0, i=1,2...N
```

同样的，对不等式约束引入拉格朗日乘子(注意先把不等式约束化成标准形式)：```$a_i \geq 0, \mu_i \geq 0$ ```, 得到：
```math
\min_{w, b , \xi} L(w, b, \xi, a, \mu) = -\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N} a_ia_jy_iy_j(x_i\cdot x_j)+\sum_{i=1}^{N} a_i
```
再对```$ \min_{w, b , \xi} L(w, b, \xi, a, \mu) $```求a的最大：
```math
\max_{a} L(w, b, \xi, a, \mu) = -\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N} a_ia_jy_iy_j(x_i\cdot x_j)+\sum_{i=1}^{N} a_i
```

**得到的对偶形式**：
```math
\min_{a}  \frac{1}{2} \sum_{i=1}^{N}\sum_{j=1}^{N} a_ia_jy_iy_j(x_i\cdot x_j)-\sum_{i=1}^{N} a_i
```


``` $s.t$ ```
```math
\sum_{i=1}^{N} a_iy_i=0

0 \leq a_i \leq C, i=1,2...N
```
先求得```$a*=(a_1,...,a_N) $ ```,再计算向量w:
```math
w = \sum_{i=1}^{N} a_iy_ix_i
```
并选择一个```$a$ ```的正分量```$ 0 \leq a_j \leq C $ ```,计算:
```math
b=y_j-\sum_{i=1}^{N} a_iy_i(x_i \cdot x_j)
```
这里的b不唯一，可以选择取所有满足条件的样本点的平均值。

针对问题里面的不等式约束，对偶问题求解满足的**KKT条件**：
```math
a_i(y_i(w\cdot x_i +b)-1+\xi_i) =0

y_i(w\cdot x_i +b)-1+\xi_i \geq 0 

\xi_i \geq 0

\mu_i \geq 0

\mu_i\xi_i = 0

a_i \geq 0  ,i=1,2,...N
```
线性支持向量机可以表示为合页损失函数的形式：
```math

\xi_i = 1-y_i(w\cdot x_i+b)

f(n) = \begin{cases}
        \xi_i,  & \text{$\xi_i > 0$} \\
        0, & \text{$\xi_i \leq 0$}
        \end{cases}
```

#### 3.非线性支持向量机：
假设样本空间非线性可分，可以通过非线性变换将它转化为某个高维特征空间的线性分类问题。一般使用核函数来做非线性变换和软间隔最大化，学习非线性支持向量机。

**核函数：**
当样本空间为欧式空间或者离散的集合、特征空间为希尔伯特空间时，核函数表示将输入从样本空间映射到特征空间得到的特征向量之间的内积。通过核函数可以学习非线性支持向量机，等价于隐式的在高维的特征空间中学习线性支持向量机。

常见的核函数：
0. 线性核
```math
k(x_i, x_j) = x_i^Tx_j
```

1. 多项式核函数
```math
k(x_i, x_j) = (x_i^T x_j)^d
```
对应的分类器为一个d次多项式分类器，分类决策函数为：
```math
f(x) = sign{\sum_{i=1}^{N_s}a_iy_i(x_i x_j)^d+b}
```
2. 高斯核函数
```math
k(x_i, x_j) = exp(-\frac{\parallel x_i-x_j \parallel ^2}{2\delta^2})
```
```$\delta $ ``` 为高斯核的带宽，分类决策函数为：
```math
f(x) = sign{\sum_{i=1}^{N_s}a_iy_iexp(-\frac{\parallel x_i-x_j \parallel ^2}{2\delta^2})+b}
```


#### SMO算法求解凸二次规划问题
SMO算法的基本思路是：如果所有变量解都满足此优化问题的KKT条件，那么这个最优化问题的解就得到了。SMO算法选择两个变量，固定其他变量，针对这两个变量构建一个二次规划问题。SMO将原问题不断分解为子问题并对子问题求解，子问题可以通过解析方法求解，可以大大提高算法计算速度。

假设当前需要求解的凸二次优化问题如下：
```math
\min_{a}  \frac{1}{2} \sum_{i=1}^{N}\sum_{j=1}^{N} a_ia_jy_iy_j(x_i\cdot x_j)-\sum_{i=1}^{N} a_i
```
``` $s.t$ ```
```math
\sum_{i=1}^{N} a_iy_i=0

0 \leq a_i \leq C, i=1,2...N
```
选择```$a_1, a_2 $ ```为两个变量，```$a_3, a_4,...a_n $ ```为固定，有
```math
a_1y_1 +\sum_{i=2}^{n}a_iy_i = 0

a_1= -y_1 \sum_{i=2}^{n}a_iy_i
```
a2确定了，a1也就确定了。

选择```$a_1, a_2 $ ```为待优化变量，原始的优化目标可以写为新的二次规划目标函数：
```math
w(a_1, a_2) = \frac{1}{2}K_{11}a_1^2+\frac{1}{2}_{22}a_2^2+y_1y_2K_{12}a_1a_2-(a_1+a_2)+y_1a_1 \sum_{i=3}^{N} y_ia_iK_{i1}+ y_2a_2\sum_{i=3}^{N}y_ia_iK_{i2}+ const

```
``` s.t. ```

```math
a_1y_1+a_2y_2 = -\sum_{i=3}^{N} y_ia_i = \zeta

0 \leq a_i \leq C, i=1, 2
```
这里有
```math
y_i*y_i = 1
```
把``` $ a_1 $```用``` $ a_2$ ```和``` $ y_1$```表示，并带入上诉公式，对``` $ a_2 $```取导等于0，求极值，得到``` $ a_2 $```的优化公式：
```math
a_2^{new,unc} = a_2^{old}+\frac{y_2(E_1-E_2)}{\eta}

\eta = K_{11}+K_{22}-2K_{12} = \parallel \Theta(x_1)-\Theta(x_2) \parallel ^2
```

当然这里的情况是为对a2进行不等式约束的情况，a1, a2需要满足一些不等式约束才能求解成立：
```math
a_1y_1+a_2y_2 = -\sum_{i=3}^{N} y_ia_i = \zeta

0 \leq a_i \leq C, i=1, 2
```
取值范围约束：
```math
L \leq a_2 \leq H
```
不等式``` $y_1 != y_2$ ```时：
![](https://note.youdao.com/yws/api/personal/file/WEBe3b6723aadee69403b1936022c5f01ac?method=getImage&version=4007&cstk=vR720X_b)

```math
a_1^{old} = a_2^{old}+\zeta

0 \leq a_2 \leq C

```
得到：
```math
0 \leq a_2 \leq C

-\zeta \leq a_2 \leq c-\zeta

L = max(0, a_2^{old}-a_1^{old})

H = min(C, C+a_2^{old}-a_1^{old})
```
同样的，``` $y_1 = y_2$ ```时，
![](https://note.youdao.com/yws/api/personal/file/WEBc41e8cd2c0a7e9e9754be9e3380055a7?method=getImage&version=4008&cstk=vR720X_b)

```math
a_1^{old} =- a_2^{old}+\zeta

0 \leq a_2 \leq C

\zeta -c \leq a_2 \leq \zeta
```
得到：

```math

L = max(0, a_2^{old}+a_1^{old} - C)

H = min(C, a_2^{old}+a_1^{old})
```

在已知``` $ y_1$ ``` , ``` $y_2$```情况下，a_2进行剪辑：
```math
a_{2}^{new}= \begin{cases} H, & a_2^{new, unc} > H \\a_2^{new, unc} , & \text{$L \leq a_2^{new, unc} \leq H$} \\ L, & \text{ $ a_2^{new, unc} < L $ }
\end{cases}
```

```math
a_1^{new} = a_1^{old} + y_1y_2(a_2^{old} - a_2^{new})
```

SMO算法在每个字问题中选择两个变量优化，其中至少一个变量是违法KKT条件的。

选择第一个变量的过程被称为外层循环，选择第二层变量被称为内层循环。选择第一个变量时，尽量的选择违背KKT条件最严重的样本点，KKT条件：
```math
y_i f(x_i) = \begin{cases} \geq 1, & a_i = 0 \\1 , & 0 < a_i <1 \\ \leq 1,&  a_i = C
\end{cases}
```
假设第一个变量已经找到情况下，选择第二个变量的标准是希望能使a2有足够大的变化。
```$a_2^{new} $``` 是依赖于``` $|E_1 -E_2 | $```的。也就是让``` $|E_1 -E_2 | $```的绝对值足够大，在计算的过程中，把所有的``` $E_i$ ```保存在一个列表里面。如果内层循环找不到合适的a2,就采用以下的启发式搜素方法：遍历在间隔边界上的支持向量点，依次将其对应的变量作为a2的适用，知道目标函数有足够的下降，若找不到合适的a2，则遍历训练数据集，若还找不到合适的a2,则放弃当前a2,通过外层循环选择另外一个a1.

计算阈值b和差值``` $E_i$ ```，每次完成两个变量的更新后，都需要重新计算阈值b.

当``` $0 < a_1^{new} < C $ ```:
```math
E_i = g(x_i)- y_i = (\sum_{j=1}^{N}a_jy_jK(x_j, x_i)+b)-y_i

i = 1, 2

b_1^{new} = -E_1-y_1K_{11}(a_1^{new}-a_1^{old}) - y_2K_{21}(a_2^{new}- a_2^{old})+b^{old}
```

当``` $0 < a_2^{new} < C $ ```:
```math
b_2^{new} = -E_2-y_1K_{12}(a_1^{new}-a_1^{old}) - y_2K_{22}(a_2^{new}- a_2^{old})+b^{old}
```
如果```$ a_1^{new}, a_2^{new} $``` 同时满足``` $0 < a_i^{new} < C $ ```,
```$b_1^{new} = b_2^{new}$ ```;
如果```$ a_1^{new}, a_2^{new} $```是0或者C,那么
```$b^{new} = \frac{1}{2}( b_1^{new}+ b_2^{new}) $```

每次完成两个变量优化后，需要更新对应的``` $E_i $ ```值，并保存到列表里面：
```math
E_i^{new} = \sum_{S} y_ja_jK(x_i, x_j)+b^{new}-y_i
```
==S为所有支持向量```$x_j$```的集合==???

SMO算法的**终止条件**可以为KKT条件对所有向量均满足，或者目标函数增长率小于某个阈值，即：
```math
\frac{w(a_{t+1}) -w(a_{t})}{w(a_{t})} < \epsilon

```

#### SVR
![svr](https://note.youdao.com/yws/api/personal/file/WEB3d9a9d7916f5bc7b0827d2b0a2b01de0?method=getImage&version=3520&cstk=vR720X_b)

这里考虑的是支持向量的回归模型，对于样本数据```$(x_i,y_i)$ ```， 传统的回归模型直接基于模型输出f(x)与真实值y之间的误差损失来学习模型，当且仅当f(x)与y完全相同时，误差才为0.这里面考虑的支持向量回归模型假设容忍模型f(x)与y之间最多有```$\epsilon $ ``` 的偏差，凡是落入到f(x)的 ( -```$\epsilon$ ```, +```$\epsilon $ ``` )的区间内的预测都被认为是正确的预测，预测的值超过了这个区间才计算损失。
学习的预测模型为：
```math
f(x) = w^T\cdot x+b
```
SVR问题定义为：
```math
\min_{w, b} \frac{1}{2} \parallel w \parallel^2 + C \sum_{i=1}^{m}\phi_\epsilon(f(x_i)-y_i)
```
其中，C为正则化常数，``` $ \phi_\epsilon $ ``` 为不敏感损失函数：
```math
\phi_\epsilon(z) = \begin{cases} 0, & \text {if $\mid z \mid \leq \epsilon $ } \\ \mid z \mid - \epsilon, & \text{otherwise} \end{cases}
```
![3](https://note.youdao.com/yws/api/personal/file/WEBa11794accfda4b492211772e0d98570d?method=getImage&version=3566&cstk=vR720X_b)

