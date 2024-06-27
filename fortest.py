import matplotlib.pyplot as plt
import numpy as np

# 示例数据
x = [36,64,100,144,196,256]
y = [11.93,13.73,15.62,18.15,20.5,22.66]
# 线性拟合
slope, intercept = np.polyfit(x, y, 1)
# 创建折线图
plt.plot(x, y, marker='o')

# 添加标题和标签

plt.xlabel(r'$h^2 \, (cm^2)$')
plt.ylabel(r'$Y^{2}h \, (s^2 \cdot cm)$')


print(f"斜率: {slope:.3f}")
print(f"截距: {intercept:.3f}")
# 显示图表
plt.show()
