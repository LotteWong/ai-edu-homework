import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def load_data():
    # 读取波士顿房价数据
    boston = load_boston()
    X, y = boston.data, boston.target.reshape(-1, 1)

    # 切分训练集和验证集
    return model_selection.train_test_split(X, y, test_size=0.3, random_state=0)


class LinearRegression(object):
    # 加载参数
    def __init__(self, eta, epo, n):
        self.eta = eta  # 学习速率
        self.epo = epo  # 迭代次数
        self.theta = np.zeros((n, 1))  # 权重偏置

    # 线性模型
    def model(self, X):
        y = X.dot(self.theta)
        return y

    # 损失函数：用均方差
    def loss(self, X, y):
        y_true = y
        y_pred = self.model(X)
        return np.mean((y_true - y_pred) ** 2) / 2

    # 更新参数：梯度下降
    def update(self, X, y):
        m = X.shape[0]  # 标签个数
        dTheta = X.T.dot(self.model(X) - y) / m

        self.theta = self.theta - self.eta * dTheta

    # 模型训练
    def train(self, X, y):
        iter_list = np.array([i for i in range(0, self.epo + 1)])  # 横坐标迭代数
        loss_list = [self.loss(X, y)]  # 纵坐标损失值

        # 梯度下降调整w和b
        for each in range(self.epo):
            self.update(X, y)
            loss_list.append(self.loss(X, y))

        return iter_list, loss_list

    # 模型预测
    def predict(self, X):
        # 用线性模型返回y
        return self.model(X)

    # 数据的可视化
    def visualize(self, iter_list, loss_list):
        plt.figure('Linear Regression Training with gradient decent')
        plt.title('Linear Regression Training with gradient decent')
        ax = plt.gca()
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.plot(iter_list, loss_list, color='r', linewidth=1, alpha=0.6)
        plt.show()


if __name__ == '__main__':
    # 读取数据
    X_train, X_valid, y_train, y_valid = load_data()
    m = X_train.shape[0]  # 标签个数
    n = X_train.shape[1]  # 特征个数

    # 处理数据
    X_StandardScaler = StandardScaler()
    y_StandardScaler = StandardScaler()
    X_train = X_StandardScaler.fit_transform(X_train)
    X_valid = X_StandardScaler.transform(X_valid)
    y_train = y_StandardScaler.fit_transform(y_train)
    y_valid = y_StandardScaler.transform(y_valid)

    # 加载模型
    eta = 0.1  # 学习速率
    epo = 100  # 迭代次数
    rgm = LinearRegression(eta, epo, n)

    # 未训练结果
    print("训练前训练集损失值：", rgm.loss(X_train, y_train))
    print("训练前验证集损失值：", rgm.loss(X_valid, y_valid))

    # 训练集结果
    iter_list, loss_list = rgm.train(X_train, y_train)
    rgm.visualize(iter_list, loss_list)
    print("训练后训练集损失值：", rgm.loss(X_train, y_train))

    # 验证集结果
    print("训练后验证集损失值：", rgm.loss(X_valid, y_valid))
    y_pred = y_StandardScaler.inverse_transform(rgm.predict(X_valid))
    print("训练后验证集的结果：")
    print(y_pred)

    # 模型的评估
    print("mean squared error", mean_squared_error(y_valid, y_pred))
