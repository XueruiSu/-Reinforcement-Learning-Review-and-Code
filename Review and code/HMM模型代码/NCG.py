import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NCG():
    def __init__(self):
        # 给定相关种类参数：
        self.N1 = 4 # 观测状态种数
        self.N2 = 4 # 隐状态种数
        self.Y_dim = 3
        self.W_dim = 4
        # 模型参数：
        self.sum_dim = self.N2*self.N1 + self.N2*self.N1*self.Y_dim + self.N2 + self.N2*self.W_dim
        self.sigma0 = np.random.normal(0, 1, (self.sum_dim, 1)) # 初始参数服从N(0,1)的正态分布
        # self.sigma0 = np.ones((self.sum_dim, 1))*0.21
        # 参数分割：
        self.mu = self.sigma0[:self.N2 * self.N1, 0].reshape((self.N1, self.N2))
        beta = self.sigma0[self.N2 * self.N1:self.N2 * self.N1 + self.N2 * self.N1 * self.Y_dim, 0]
        self.beta = beta.reshape((self.N1, self.N2, self.Y_dim))
        N1 = self.N2 * self.N1 + self.N2 * self.N1 * self.Y_dim
        N2 = self.N2 * self.N1 + self.N2 * self.N1 * self.Y_dim + self.N2
        self.gamma = self.sigma0[N1:N2, 0].reshape((self.N2, 1))
        self.alpha = self.sigma0[N2:, 0].reshape((self.W_dim, self.N2))
        # self.Pij = np.random.uniform(0, 1, (self.N1, self.N1))
        # 需要给定下面三个参数：
        self.W = np.array([[31.167], [34.125], [26.667], [27.271]])
        Pij = pd.read_excel('./4观测转换频率.xlsx', sheet_name="Sheet1", header=None)
        self.Pij = Pij.values
        print(self.Pij, self.Pij.shape)
        Y_all = pd.read_excel('./4观测转换频率 - 副本.xlsx', sheet_name="Sheet1", header=None)
        self.Y_all = Y_all.values
        print(self.Y_all, self.Y_all.shape)
        # self.Pij[0, 0] = 1
        self.Pij = self.Pij / self.Pij.max(axis=1) # 行和置为1
        # 算法参数：
        self.kmax = 200
        self.eps = 1e-6

    # P(Yi, Yj)概率计算函数
    def funPij(self, mu, beta, gamma, alpha, Yi, Yj, W, i, j):
        Pij = 0
        for K1 in range(self.N2):
            for K2 in range(self.N2):
                P_Sk1_Yi_1 = np.exp(mu[i, K1] - beta[i, K1, :].reshape((1, self.Y_dim))@Yi)
                P_Sk1_Yi_2 = 0
                for j1 in range(self.N1):
                    P_Sk1_Yi_2 = P_Sk1_Yi_2+np.exp(mu[j1, K1]-beta[j1, K1, :].reshape((1,self.Y_dim))@self.Y_all[j1,:])
                P_Sk1_Yi = P_Sk1_Yi_1 / P_Sk1_Yi_2

                P_Sk2_Sk1_1 = np.exp(gamma[K2, 0] - alpha[:, K1].T@W)
                P_Sk2_Sk1_2 = 0
                for j1 in range(self.N2):
                    P_Sk2_Sk1_2 = P_Sk2_Sk1_2 + np.exp(gamma[j1, 0] - alpha[:, K1].T @ W)
                P_Sk2_Sk1 = P_Sk2_Sk1_1 / P_Sk2_Sk1_2

                P_Sk2_Yi_1 = np.exp(mu[j, K2] - beta[j, K2, :].reshape((1, self.Y_dim)) @ Yj)
                P_Sk2_Yi_2 = 0
                for j1 in range(self.N1):
                    P_Sk2_Yi_2 = P_Sk2_Yi_2+np.exp(mu[j1, K2]-beta[j1, K2, :].reshape((1,self.Y_dim))@self.Y_all[j1,:])
                P_Sk2_Yi = P_Sk2_Yi_1 / P_Sk2_Yi_2

                Pij = Pij + P_Sk1_Yi*P_Sk2_Sk1*P_Sk2_Yi
        return Pij

    def P_Y_S_fun(self, mu, beta, gamma, alpha):
        P_Y_S = np.zeros((self.N1, self.N2))
        for i in range(self.N1):
            for j in range(self.N2):
                P_Yi_Sj_1 = np.exp(mu[i, j] - beta[i, j, :].reshape((1, self.Y_dim)) @ self.Y_all[i, :])
                P_Yi_Sj_2 = 0
                for j1 in range(self.N1):
                    P_Yi_Sj_2 += np.exp(mu[j1, j] - beta[j1, j, :].reshape((1, self.Y_dim)) @ self.Y_all[j1, :])
                P_Y_S[i, j] = P_Yi_Sj_1 / P_Yi_Sj_2
        return P_Y_S

    # P(Yi, Yj)梯度计算函数
    def gfunPij(self, mu, beta, gamma, alpha, Yi, Yj, W, i, j):
        g_mu = mu - mu
        g_beta = beta - beta
        g_gamma = gamma - gamma
        g_alpha = alpha - alpha

        for K1 in range(self.N2):
            for K2 in range(self.N2):
                P1 = np.exp(mu[i, K1] - beta[i, K1, :].reshape((1, self.Y_dim))@Yi)
                P2 = 0
                for j1 in range(self.N1):
                    P2 = P2 + np.exp(mu[j1, K1] - beta[j1, K1, :].reshape((1, self.Y_dim)) @ self.Y_all[j1, :])
                P_Sk1_Yi = P1 * (P2-P1) / P2 / P2

                P_Sk2_Sk1_1 = np.exp(gamma[K2, 0] - alpha[:, K1].T @ W)
                P_Sk2_Sk1_2 = 0
                for j1 in range(self.N2):
                    P_Sk2_Sk1_2 = P_Sk2_Sk1_2 + np.exp(gamma[j1, 0] - alpha[:, K1].T @ W)
                P_Sk2_Sk1 = P_Sk2_Sk1_1 / P_Sk2_Sk1_2

                P_Sk2_Yj_1 = np.exp(mu[j, K2] - beta[j, K2, :].reshape((1, self.Y_dim)) @ Yj)
                P_Sk2_Yj_2 = 0
                for j1 in range(self.N1):
                    P_Sk2_Yj_2 = P_Sk2_Yj_2+np.exp(mu[j1, K2]-beta[j1, K2, :].reshape((1,self.Y_dim))@self.Y_all[j1,:])
                P_Sk2_Yj = P_Sk2_Yj_1 / P_Sk2_Yj_2

                for j1 in range(self.N1):
                    if j1 == i:
                        g_mu[i, K1] = g_mu[i, K1] + P_Sk1_Yi * P_Sk2_Sk1 * P_Sk2_Yj
                        g_beta[i, K1, :] = g_beta[i, K1, :] + (-Yi)*P_Sk1_Yi * P_Sk2_Sk1 * P_Sk2_Yj
                    else:
                        P3 = np.exp(mu[j1, K1] - beta[j1, K1, :].reshape((1, self.Y_dim))@self.Y_all[j1, :])
                        P8 = (-P3*P1/P2/P2) * P_Sk2_Sk1 * P_Sk2_Yj
                        g_mu[j1, K1] = g_mu[j1, K1] + P8
                        g_beta[j1, K1, :] = g_beta[j1, K1, :] + (self.Y_all[j1, :])*(P3*P1/P2/P2)*P_Sk2_Sk1*P_Sk2_Yj

                P_Sk2_Yj = P_Sk2_Yj_1 * (P_Sk2_Yj_2-P_Sk2_Yj_1) / P_Sk2_Yj_2 / P_Sk2_Yj_2
                P_Sk1_Yi = P1 / P2

                for j1 in range(self.N1):
                    if j1 == j:
                        g_mu[j, K2] = g_mu[j, K2] + P_Sk1_Yi * P_Sk2_Sk1 * P_Sk2_Yj
                        g_beta[j, K2, :] = g_beta[j, K2, :] + (-Yj)*P_Sk1_Yi * P_Sk2_Sk1 * P_Sk2_Yj
                    else:
                        P3 = np.exp(mu[j1, K2] - beta[j1, K2, :].reshape((1, self.Y_dim))@self.Y_all[j1, :])
                        P8 = P_Sk1_Yi * P_Sk2_Sk1 * (-P3*P_Sk2_Yj_1/P_Sk2_Yj_2/P_Sk2_Yj_2)
                        g_mu[j1, K2] = g_mu[j1, K2] + P8
                        g_beta[j1, K2, :] = g_beta[j1, K2, :] + (-self.Y_all[j1, :])*P8

                P_Sk2_Yj = P_Sk2_Yj_1 / P_Sk2_Yj_2
                P_Sk2_Sk1 = P_Sk2_Sk1_1 * (P_Sk2_Sk1_2-P_Sk2_Sk1_1) / P_Sk2_Sk1_2 / P_Sk2_Sk1_2

                for j1 in range(self.N2):
                    if j1 == K2:
                        g_gamma[K2, 0] = g_gamma[K2, 0] + P_Sk1_Yi * P_Sk2_Sk1 * P_Sk2_Yj
                        P6 = W - W
                        for j2 in range(self.N2):
                            P6 = P6 + (-W)*np.exp(gamma[j2, 0] - alpha[:, K1].T @ W)
                        P7 = ((-W) * P_Sk2_Sk1_1) @ P_Sk2_Sk1_2
                        P6 = P6 @ P_Sk2_Sk1_1
                        g_alpha[:, K1] = g_alpha[:, K1] + ((P7-P6) / P_Sk2_Sk1_2[0] / P_Sk2_Sk1_2[0])*P_Sk1_Yi*P_Sk2_Yj
                    else:
                        P4 = np.exp(gamma[j1, 0] - alpha[:, K1].T @ W)
                        P5 = (-P4 * P_Sk2_Sk1_1 / P_Sk2_Sk1_2 / P_Sk2_Sk1_2) * P_Sk1_Yi * P_Sk2_Yj
                        g_gamma[j1, 0] = g_gamma[j1, 0] + P5
        return g_mu, g_beta, g_gamma, g_alpha

    # 参数粘贴函数
    def SigmaShape(self, mu, beta, gamma, alpha):
        sigma1 = mu.reshape(-1, 1)
        sigma2 = beta.reshape(-1, 1)
        sigma3 = gamma.reshape(-1, 1)
        sigma4 = alpha.reshape(-1, 1)
        return np.concatenate((sigma1, sigma2, sigma3, sigma4), axis=0)

    # 参数分割函数
    def SigmaReshape(self, sigma):
        mu = sigma[:self.N2 * self.N1, 0].reshape((self.N1, self.N2))
        beta = sigma[self.N2 * self.N1:self.N2 * self.N1 + self.N2 * self.N1 * self.Y_dim, 0]
        beta = beta.reshape((self.N1, self.N2, self.Y_dim))
        N1 = self.N2 * self.N1 + self.N2 * self.N1 * self.Y_dim
        N2 = self.N2 * self.N1 + self.N2 * self.N1 * self.Y_dim + self.N2
        gamma = sigma[N1:N2, 0].reshape((self.N2, 1))
        alpha = sigma[N2:, 0].reshape((self.W_dim, self.N2))
        return mu, beta, gamma, alpha

    # 损失函数（算法目标函数）
    def fun(self, sigma, Y, W, Pij):
        mu, beta, gamma, alpha = self.SigmaReshape(sigma)
        loss = 0
        for i in range(self.N1):
            for j in range(self.N1):
                pij = self.funPij(mu, beta, gamma, alpha, Y[i, :].T, Y[j, :].T, W, i, j)
                loss = loss + np.power(pij-Pij[i, j], 2)
        return loss

    # 损失函数梯度
    def gfun(self, sigma, Y, W, Pij):
        mu, beta, gamma, alpha = self.SigmaReshape(sigma)
        g_mu = self.mu - self.mu
        # print(g_mu)
        g_beta = self.beta - self.beta
        g_gamma = self.gamma - self.gamma
        g_alpha = self.alpha - self.alpha
        for i in range(self.N1):
            for j in range(self.N1):
                g_mu_n, g_beta_n, g_gamma_n, g_alpha_n = self.gfunPij(mu,beta,gamma,alpha,Y[i,:].T, Y[j, :].T, W, i, j)
                pij = self.funPij(mu, beta, gamma, alpha, Y[i, :].T, Y[j, :].T, W, i, j)
                g_mu = g_mu + 2*(pij-Pij[i, j])*g_mu_n
                g_beta = g_beta + 2*(pij-Pij[i, j])*g_beta_n
                g_gamma = g_gamma + 2*(pij-Pij[i, j])*g_gamma_n
                g_alpha = g_alpha + 2*(pij-Pij[i, j])*g_alpha_n
        sigma = self.SigmaShape(g_mu, g_beta, g_gamma, g_alpha)
        return sigma

    # nonlinear_cg--FR g_k^Tg_k/(g_{k-1}^Tg_{k-1})
    def FR(self, g1, g2):
        print("调用FR公式")
        return np.dot(g2.T, g2) / np.dot(g1.T, g1)

    # nonlinear_cg--PR
    def PR(self, g1, g2):
        print("调用PR公式")
        return max(0, np.dot(g2.T, (g2 - g1)) / (np.dot(g1.T, g1)))

    # x 当前迭代点 d 迭代方向 g当前迭代点的梯度 f 目标函数在当前点的函数值
    # Armijo步长规则
    def Armijo(self, sigma0, Y, W, Pij, d, g, f):
        alpha1 = 1
        sigma = 1e-3
        print("调用Armijo步长规则")
        for i in range(20):
            sigma_n = sigma0 + alpha1*d
            fn = self.fun(sigma_n, Y, W, Pij)
            if fn < f + sigma * alpha1 * np.dot(g.T, d):
                break
            alpha1 = alpha1 * 0.1
        return alpha1

    # 非线性共轭梯度算法框架
    def nonlinear_cg(self, sigma0, Y, W, Pij, searching, beta_para):
        print('---------BEGIN---------')
        start = time.time()
        x_old = sigma0
        g_old = self.gfun(x_old, Y, W, Pij)
        # print("1:", g_old)
        d_old = -g_old
        f_list = []
        for k in range(self.kmax):
            f_old = self.fun(x_old, Y, W, Pij)
            print("当前残差函数值:", f_old)
            if k % 100 == 0:
                print(f_old)
            f_list.append(f_old[0])
            g_old_norm = np.linalg.norm(g_old, 2)
            print('当前迭代第{}步,当前精度值{:.8f}'.format(k, g_old_norm))
            if g_old_norm < self.eps:
                break
            alpha = searching(x_old, Y, W, Pij, d_old, g_old, f_old)
            # 更新 x
            x_new = x_old + alpha * d_old
            # 计算当前迭代点的梯度
            g_new = self.gfun(x_new, Y, W, Pij)
            beta = beta_para(g_old, g_new)
            d_new = -g_new + beta * d_old

            #更新参数
            d_old = d_new
            g_old = g_new
            x_old = x_new
        end = time.time()
        time_use = end - start
        return x_new, f_list, k, time_use

    # 算法主框架：
    def main(self):
        nlcg_x1, nlcg_f1, nlcg_k1, nlcg_time1 = self.nonlinear_cg(self.sigma0, self.Y_all, self.W, self.Pij, \
                                                                  self.Armijo, self.FR)

        print("最优参数为：", self.SigmaReshape(nlcg_x1))
        mu, beta, gamma, alpha = self.SigmaReshape(nlcg_x1)
        print("观测概率为：", self.P_Y_S_fun(mu, beta, gamma, alpha))
        data_df = pd.DataFrame(self.P_Y_S_fun(mu, beta, gamma, alpha))  # 关键1，将ndarray格式转换为DataFrame
        # 更改表的索引
        data_df.columns = ['S1', 'S2', 'S3', 'S4']  # 将第一行的0,1,2,...,9变成A,B,C,...,J
        data_df.index = ['Y1', 'Y2', 'Y3', 'Y4']

        # 将文件写入excel表格中
        writer = pd.ExcelWriter('P_Y_S.xlsx')  # 关键2，创建名称为hhh的excel表格
        data_df.to_excel(writer, 'page_1',
                         float_format='%.5f')  # 关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
        writer.save()
        print('“Armijo&FR”最优的目标函数值为{:.2f},使用时间: {:.2f}s'.format(nlcg_f1[-1], nlcg_time1))

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
        plt.rcParams['axes.unicode_minus'] = False
        plt.plot(range(nlcg_k1 + 1), nlcg_f1, 'r', markersize=2, linewidth=2, label='强Wolfe+FR')
        plt.legend()
        plt.xlabel("迭代次数")
        plt.ylabel("目标函数值")
        plt.title("非线性共轭梯度迭代过程")
        plt.savefig("./figure/非线性共轭梯度法rand.png")
        plt.show()

if __name__ == '__main__':
    NCG1 = NCG()
    NCG1.main()







