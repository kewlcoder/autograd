import autograd.numpy as np
from autograd import grad
from autograd.test_util import check_grads


def sigmoid(x):
    
    # This approximation of sigmoid performs better than sigmoid
    # return 0.5*(np.tanh(x) + 1)
    
    # The below is same formula as above when both expanded
    # return 1/( 1 + np.e**(-2*x))

    # below is the sigmoid function 
    return 1/( 1 + np.e**x)

def predict(w, x):
    wx = np.dot(x, w)
    # print('wx.shape = ', wx.shape)
    return sigmoid(wx)

'''
'''
def compute_loss(w, x, y):
    """
    Note - 
    "weights" should necessarily be made the first argument of the 
    loss computation function.
    Further exxploration - 
    The second argument of grad() can be used to specify the argument of subject function(compute_loss) on which the 
    gradient needs to be evaluated. This second arg can be a tuple. (1,2,3) implies - calc gradient over arguments 
    1,2 and 3 of the subject func.
    Further, we can always specify all the variables for which the gradient has to be calculated in the computational
    graph as a vector(as has been done for "weights" variables in this example).

    Args:
        w ([type]): [description]
        x ([type]): [description]
        y ([type]): [description]

    Returns:
        [type]: [description]
    """        
    preds = predict(w, x)
    probabilities = preds*y + (1-preds)*(1-y)
    # print('probabilities.shape = ', probabilities.shape)
    # print("a")
    return -np.sum( np.log(probabilities) )


# Build a toy dataset.
inputs = np.array([[0.52, 1.12,  0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]])
labels = np.array([True, True, False, True])


gradient_func = grad(compute_loss)

'''
This shape(3,) for weights1 is very important and required as otherwise 
the matrix multiplication is inferred differently in the predict &
compute_loss funcs.
'''
weights1 = np.random.normal(0, 0.3, size=(3,))
# weights1 = np.random.normal(1, 0.1, size=(3, 1))
# weights1 = np.random.normal(1, 0, size=(3, 1))

# weights1 = weights1.reshape(weights1.shape[0])

check_grads(compute_loss, modes=['rev'])(weights1, inputs, labels)

weights2 = np.array([0.0, 0.0, 0.0])
# weights2 = np.array([1.0, 1.0, 1.0])


# print(weights.shape)
print(weights1.shape)
print(weights1)

print(weights2.shape)
print(weights2)
# print(np.array([0.0, 0.0, 0.0]).shape)


print("starting loss for weights1 = ", compute_loss(weights1, inputs, labels))
print("starting loss for weights2 = ", compute_loss(weights2, inputs, labels))

for i in range(100):
    weights1 -= gradient_func(weights1, inputs, labels) * 0.1
    weights2 -= gradient_func(weights2, inputs, labels) * 0.1
    # print('b')
    # print('weights1.shape', weights1.shape)
    # print('tmp = ', tmp)
    # print('tmp.shape', tmp.shape)

print(weights1)
print(weights2)

print("After-Training loss - weights1 = ", compute_loss(weights1, inputs, labels) )
print("After-Training loss - weights2 = ", compute_loss(weights2, inputs, labels) )