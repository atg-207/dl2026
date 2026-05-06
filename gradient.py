def f(x):
    return x**2

def df(x):
    return 2*x

def gradient_descent(x_start, learning_rate, threshold):
    x = x_start
    time = 0

    print(f"{'Time':<10} | {'x':<10} | {'f(x)':<10}")
    print("-" * 35)

    while f(x) > threshold:
        time += 1

        grad = df(x)
        x = x - learning_rate * grad

        print(f"{time:<10} | {x:<10.4f} | {f(x):<10.4f}")

        if time >= 100:
            break

    return x

if __name__ == "__main__":
    gradient_descent(x_start=10, learning_rate=0.1, threshold=0.01)
