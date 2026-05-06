def f(x):
    return x**2

def df(x):
    return 2*x

def gradient_descent(x_start, learning_rate, epochs):
    x = x_start
    print(f"{'Step':<10} | {'x':<10} | {'f(x)':<10}")
    print("-" * 35)
    for i in range(epochs):
        grad = df(x)
        x = x - learning_rate * grad
        print(f"{i+1:<10} | {x:<10.4f} | {f(x):<10.4f}")  
    return x

if __name__ == "__main__":
    final_x = gradient_descent(x_start=10, learning_rate=0.1, epochs=10)
    print(f"The local minimum occurs at {final_x:.4f}")
