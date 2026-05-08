import csv 

def load_data(filename):
    X, Y = [], []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            X.append(float(row[0]))
            Y.append(float(row[1]))
    return X, Y

def linear_regression(X, Y, r, t):
    w0 = 0
    w1 = 1
    N = len(X)
    epoch = 0

    print(f"{'Epoch':<10} | {'w0':<10} | {'w1':<10} | {'loss':<10}")
    print("-" * 35)

    while True:
        epoch += 1
        sum_dw0 = 0
        sum_dw1 = 0
        total_loss = 0

        for i in range(N):
            y_hat = w0 + w1 * X[i]
            error = y_hat - Y[i]

            sum_dw0 += error
            sum_dw1 += error * X[i]

            total_loss += 0.5 * (error ** 2)
        
        w0 = w0 - r * sum_dw0 / N
        w1 = w1 - r * sum_dw1 / N

        avg_loss = total_loss / N

        if epoch % 100 == 0:
            print(f"{epoch:<10} | {w0:<10.4f} | {w1:<10.4f} | {avg_loss:<10.4f}")

        if avg_loss < t or epoch >= 100000:
            break

    return w0, w1

if __name__ == "__main__":
    X, Y = load_data('lr.csv')
    r = 0.0008
    t = 0.001
    w0, w1 = linear_regression(X, Y, r, t)
    print(f"w0 = {w0}, w1 = {w1}")