import matplotlib.pyplot as plt
import numpy as np

# for k
# -5.066e-06 x^4 + 0.00118 x^3 - 0.0937 x^2 + 2.829 x + 40.74
x = np.array([2, 6, 10, 12, 18, 24, 36, 48, 56, 60, 80, 90, 100])
y = np.array([39.2, 61.6, 63.8, 65.3, 65.2, 65.9, 65.5, 65.0, 64.1, 64.8, 63.0, 62.9, 61.5])

z = np.polyfit(x, y, 4)  # using 4 polynomial to fit
p = np.poly1d(z)
print(p)
plot1 = plt.plot(x, y, '*', label='real recall@1')

x2 = np.linspace(1, 100, 100)
y2 = p(x2)
plot2 = plt.plot(x2, y2, 'r', label='predict recall@1')

plt.xlabel('k value')
plt.ylabel('recall@1')
plt.legend(loc=4)
plt.title('k ~ recall@1')
plt.savefig('k_recall.pdf')
plt.show()
