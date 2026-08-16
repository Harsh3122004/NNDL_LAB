import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 40, 50, 78, 83])


m = 0
b = 0

learning_rate = 0.01
epochs = 1000
n = len(x)


for i in range(epochs):
    y_pred = m * x + b

    dm = (-2/n) * np.sum(x * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)

    m = m - learning_rate * dm
    b = b - learning_rate * db

    if i % 100 == 0:
        loss = np.mean((y - y_pred) ** 2)
        print(f"Epoch {i}: Loss = {loss:.4f}, m = {m:.4f}, b = {b:.4f}")

print("\nFinal parameters:")
print("Slope (m):", m)
print("Intercept (b):", b)


plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, m*x + b, color='red', label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()