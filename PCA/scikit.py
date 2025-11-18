import numpy as np
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

np.set_printoptions(
    precision=10,
    threshold=10000,
    suppress=True
)

# Загружаем данные и удаляем строки с пропущенными значениями
data = np.genfromtxt(
    r"Data\table.csv",
    delimiter=',',
    skip_header=1,
    usecols=list(range(1, 11))
)
data = data[~np.isnan(data).any(axis=1)]

# Стандартизируем данные
data = scale(data)

# Выполняем метод главных компонент
pca = PCA(svd_solver='full')
pca.fit(data)

print("Размерность данных:\n", data.shape, "\n")

# Вклад каждого фактора PCA
print("Вклад каждого фактора в объяснение вариации:\n",
      pca.explained_variance_ratio_, "\n")

# Накопленная вариация
var = np.round(np.cumsum(pca.explained_variance_ratio_), decimals=4)
print("Рост доли объясненной вариации с увеличением числа главных факторов:\n",
      var, "\n")

# График накопленной дисперсии
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(var) + 1), var, marker='o')
plt.ylabel('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.title('PCA: Cumulative Explained Variance')
plt.grid(True)
plt.show()
