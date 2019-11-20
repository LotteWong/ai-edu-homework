from sklearn.datasets import load_iris
from sklearn import model_selection
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


def load_data():
    # 读取鸢尾花种类数据
    iris = load_iris()
    X, y = iris.data, iris.target.reshape(-1, 1)

    # 切分训练集和验证集
    return model_selection.train_test_split(X, y, test_size=0.3, random_state=0)


if __name__ == '__main__':
    # 读取数据
    X_train, X_valid, y_train, y_valid = load_data()

    # 加载模型
    cls = linear_model.LogisticRegression(penalty='l2', solver='newton-cg', multi_class='multinomial')

    # 训练集结果
    cls.fit(X_train, y_train)

    # 验证集结果
    target_names = ['setosa', 'versicolor', 'virginica']
    y_pred = cls.predict(X_valid)
    print(classification_report(y_valid, y_pred, target_names=target_names))

    # 模型的效果
    print("accuracy score: ", accuracy_score(y_valid, y_pred))
