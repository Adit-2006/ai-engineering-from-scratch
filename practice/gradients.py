def saddle(params):
    x, y = params
    return x ** 2 - y ** 2

def saddle_gradent(params):
    x,y = params
    df_dx = 2 * x 
    df_dy = -2 * y
    return [df_dx, df_dy]
    


def rosenbrock(params):
    x, y = params
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_gradient(params):
    x, y = params
    df_dx = -2 * (1 - x) + 200 * (y - x ** 2) * (-2 * x)
    df_dy = 200 * (y - x ** 2)
    return [df_dx, df_dy]




class GradientDescent:
    def __init__(self, lr = 0.001, decay_rate = 0.1):
        self.lr = lr
        self.t = 0
        self.decay_rate = decay_rate
        


    def decay(self):
        return  self.lr * self.decay_rate ** self.t

    def step(self, params, grads):
        self.t += 1
        current_lr = self.decay()
        return [p - current_lr * g for p, g in zip(params, grads)]



class SGDMomentum:
    def __init__(self, lr=0.001, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [0.0] * len(params)
        self.velocity = [
            self.momentum * v + g
            for v, g in zip(self.velocity, grads)
        ]
        return [p - self.lr * v for p, v in zip(params, self.velocity)]

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        self.m = [
            self.beta1 * m + (1 - self.beta1) * g
            for m, g in zip(self.m, grads)
        ]
        self.v = [
            self.beta2 * v + (1 - self.beta2) * g ** 2
            for v, g in zip(self.v, grads)
        ]

        m_hat = [m / (1 - self.beta1 ** self.t) for m in self.m]
        v_hat = [v / (1 - self.beta2 ** self.t) for v in self.v]

        return [
            p - self.lr * mh / (vh ** 0.5 + self.epsilon)
            for p, mh, vh in zip(params, m_hat, v_hat)
        ]


def optimize(optimizer, func, grad_func, start, steps=5000):
    params = list(start)
    history = [params[:]]
    for _ in range(steps):
        grads = grad_func(params)
        params = optimizer.step(params, grads)
        history.append(params[:])
    return history

start = [-1.0, 1.0]

mom_lst = [0.0, 0.5, 0.9, 0.99]

#sgd_history = optimize(SGDMomentum(lr=0.0001, momentum = 0.9), rosenbrock, rosenbrock_gradient, start)
gd_without_decay = optimize(GradientDescent(lr=0.0005, decay_rate= 1.0),rosenbrock, rosenbrock_gradient, start)
gd_with_decay = optimize(GradientDescent(lr=0.0005, decay_rate= 0.999),rosenbrock, rosenbrock_gradient, start)
#adam_history = optimize(Adam(lr=0.01), rosenbrock, rosenbrock_gradient, start)

for name, history in [("GD without decay", gd_without_decay), ("GD with decay", gd_with_decay)]:
    final = history[-1]
    loss = saddle(final)
    print(f"{name:6s} -> x={final[0]:.6f}, y={final[1]:.6f}, loss={loss:.12f}")


