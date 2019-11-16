import matplotlib.pyplot as plt
import numpy as np

# for m
# -2.342 (log(x))^2 + 20.03 log(x) + 22.96
x = np.array([2, 4, 6, 8, 10, 12, 18, 24, 30, 36, 48, 64, 80])
y = np.array([35.1, 46.8, 52.2, 54.5, 56.5, 58.1, 60.7, 63.5, 63.2, 64.6, 65.3, 65.9, 66.1])

z = np.polyfit(np.log(x), y, 2)  # using 2 polynomial with log to fit
p = np.poly1d(z)
print(p)
plot1 = plt.plot(x, y, '*', label='real recall@1')

x2 = np.linspace(1, 100, 100)
y2 = p(np.log(x2))
plot2 = plt.plot(x2, y2, 'r', label='predict recall@1')

plt.xlabel('m value')
plt.ylabel('recall@1')
plt.legend(loc=4)
plt.title('m ~ recall@1')
plt.savefig('m_recall.pdf')
plt.show()
