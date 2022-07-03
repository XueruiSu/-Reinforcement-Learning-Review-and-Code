import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
DAY = 1
alpha1 = 0.02
alpha2 = 0.01

#Function generating solution
def s(gamma, moneyshuzu):
    # money为最终收益总额，strategy为DAY*2的策略矩阵，储存了每天往B/G里面投资的金额
    strategy = np.zeros((DAY, 2))
    s_money_sum = np.zeros((DAY, 3))
    rand_s = np.random.rand(DAY, 1)
    s_x = int(moneyshuzu[0])
    s_y = int(moneyshuzu[1])
    s_z = int(moneyshuzu[2])
    print('-----------------------------初始价格：', s_x, s_y, s_z)
    s_money_sum[0, 0] = s_x
    s_money_sum[0, 1] = s_y
    s_money_sum[0, 2] = s_z
    money_allsum = s_x + s_y + s_z
    for i in range(gamma.shape[0]):
        #print('如果不变：', s_x,s_y,s_z)
        #计算买进卖出：
        s_m1 = np.random.randint(-1000, 1000)
        s_m2 = np.random.randint(-1000, 1000)
        s_judge_if = 0
        s_judge_if2 = 0
        #print('shibshiwhile:', s_judge_if, s_judge_if2)
        while(s_judge_if==0 or s_judge_if2==0):
            s_m1 = np.random.randint(-1000, 1000)
            s_m2 = np.random.randint(-1000, 1000)
            #print('still')
            # if s_m1+s_m2 <= s_x and s_m1>0 and s_m1>0:
            #     break
            if s_m1+s_m2 <= s_x:
                if (s_m1<0 and (-s_m1) <= s_y) or s_m1>0:
                    #print('触发if1',s_m1, s_y, s_m2)
                    s_judge_if = 1
                    #print(1,':',s_judge_if, s_judge_if2)
                if (s_m2<0 and (-s_m2) <= s_z) or s_m2>0:
                    #print('触发if2', s_m1, s_z, s_m2)
                    s_judge_if2 = 1
                    #print(2,':',s_judge_if, s_judge_if2)
                if s_m1>=0 and s_m2>=0:
                    #print('触发if3', s_m1, s_z, s_m2)
                    s_judge_if = 1
                    s_judge_if2 = 1
            if s_judge_if == 0 or s_judge_if2 == 0:
                s_judge_if = 0
                s_judge_if2 = 0
        #print('shibshiwhile:',s_judge_if,s_judge_if2)
        s_judge_if = 0
        s_judge_if2 = 0
        strategy[i, 0] = s_m1
        strategy[i, 1] = s_m2
        #if i == 0:
            #print('在第一次预测中',s_m1, 'and', s_m2, '为策略')
        s_x = int(s_x-strategy[i, 0] - strategy[i, 1])
        s_y = int((s_y+strategy[i, 0])*(1+gamma[i, 0]) - strategy[i, 0]*alpha1)
        s_z = int((s_z+strategy[i, 1])*(1+gamma[i, 1]) - strategy[i, 1]*alpha2)
        #print('一次决策后的本金为：', s_x, s_y, s_z)
        s_money_sum[i, 0] = s_x
        s_money_sum[i, 1] = s_y
        s_money_sum[i, 2] = s_z
        money_allsum = s_x+s_y+s_z
    #print('一次s中，总策略为：',money_allsum, strategy,s_money_sum)
    return strategy, money_allsum,s_money_sum

def GA(gamma,GA_money,T,L,S,SS,GA_T_MIN,GA_alpha):
    #终止条件：当cost函数连续S次小于SS,则跳出函数
    #gamma:,money:初始金额分布,T:温度,L:每个温度迭代次数
    money_sum = np.zeros((T*L, 1))
    flag = 0
    cloc = 0
    #最优策略：
    strategy, money_sum, ce_lue_ju_zhen_DAY_3 = s(gamma, GA_money)
    strategy_best = strategy
    GA_T = T
    while (GA_T > GA_T_MIN):
        print("执行完一次降温")
        for j in range(L):
            #产生新解s':
            #print(GA_money.type)
            strategy, money_sum2, ce_lue_ju_zhen_DAY_3 = s(gamma, GA_money)

            m = money_sum - money_sum2
            #终止条件：
            if m < SS and flag<S and cloc==1:
                flag+=1
                cloc=1
            elif m<SS and flag ==0 and cloc ==0:
                flag=1
                cloc=1
            elif flag==SS and cloc ==1:
                break
            else:
                cloc=0
                flag=0
            #新解判断：
            if m<0:
                strategy_best = strategy
                money_sum = money_sum2
            else:
                p = np.exp(-m/T)
                rand = np.random.rand(1)
                if rand < p:
                    strategy_best = strategy
                    money_sum = money_sum2
        GA_T = GA_T * GA_alpha
    #print("ga计算完成,结果为：", strategy_best, money_sum)
    return strategy_best, money_sum, ce_lue_ju_zhen_DAY_3
#数据输入：
df3 = pd.read_csv("GA_pre_v01.csv",sep = ',')
df3.head()
data = df3.values
data = data[:, :2]

#输入参数：
L = 100
T = 100
S = 10
SS = 5
real_min = 100
real_alpha = 0.9

# for j in range(10):
#     DAY = j + 20
#     money = [1000, 0, 0]
#     money = np.array(money)
#     # money:
#     last2151_matrix = np.zeros((DAY, 3))
#     last2151_matrix[DAY - 1, :] = money
#     print("开始计算,初始本金分布为：", last2151_matrix[DAY - 1, :])
#     real_money_sum_history = np.zeros((data.shape[0] - DAY, 1))
#     print(real_money_sum_history.shape)
#     for i in range(int(data.shape[0]/DAY)-1):
#         # gama:
#         gamma = data[i*DAY:i*DAY+DAY]
#         print(gamma.shape)
#         real_strategy_best, real_money_sum, last2151_matrix = GA(gamma, last2151_matrix[DAY-1,:], T, L, S, SS,real_min,real_alpha)
#         print("计算完成。收益日期：", i*DAY, "，结果为：", real_strategy_best, real_money_sum, last2151_matrix)
#         real_money_sum_history[i] = real_money_sum
#         if i/20 == 0:
#             np.savetxt('ipredict' + str(j) + '+5days.txt', real_money_sum_history)
DAY = 1800
money = [1000, 0, 0]
money = np.array(money)
# money:
last2151_matrix = np.zeros((DAY, 3))
last2151_matrix[DAY - 1, :] = money
print("开始计算,初始本金分布为：", last2151_matrix[DAY - 1, :])
real_money_sum_history = np.zeros((data.shape[0] - DAY, 1))
print(real_money_sum_history.shape)
gamma = data[0:DAY]
real_strategy_best, real_money_sum, last2151_matrix = GA(gamma, last2151_matrix[DAY-1,:], T, L, S, SS,real_min,real_alpha)
    # print(i)
    # print("输出文件至：", j)
    # np.savetxt('predict' + str(j) + '+5days.txt', real_money_sum_history)
    # print("last计算完成,结果为：", real_strategy_best, real_money_sum, last2151_matrix)
np.savetxt('predict_global+5days.txt', real_money_sum_history)
print("last计算完成,结果为：", real_strategy_best, real_money_sum, last2151_matrix)




