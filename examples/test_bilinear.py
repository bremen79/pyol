import matplotlib.pyplot as plt
import numpy as np
import pyol

d=2
A=-np.array([[1, -1/5], [0,1/5]])

bsp_array = [pyol.BilinearSaddlePoint(A,pyol.EntropyOptimisticFTRL(d, lr_handler=pyol.LearningRateConstant(eta0=.5), hint_handler=pyol.HintPreviousGrad(d)), pyol.EntropyOptimisticFTRL(d, lr_handler=pyol.LearningRateConstant(eta0=.5), hint_handler=pyol.HintPreviousGrad(d))),
             pyol.BilinearSaddlePoint(A,pyol.EntropyOptimisticFTRL(d, lr_handler=pyol.LearningRateConstant(eta0=.1), hint_handler=None), pyol.EntropyOptimisticFTRL(d, lr_handler=pyol.LearningRateConstant(eta0=.1), hint_handler=None))
             ]

num_alg = len(bsp_array)

T=1000

sum_x = np.zeros([d,num_alg])
sum_y = np.zeros([d,num_alg])
array_x = np.zeros([d,T,num_alg])
array_y = np.zeros([d,T,num_alg])

gaps_average = np.zeros([T,num_alg])
gaps_last = np.zeros([T,num_alg])

for i in range(T):    
    for j,alg in enumerate(bsp_array):
        (x,y,grad_x, grad_y)=alg.update()
        
        sum_x[:,j] += x
        sum_y[:,j] += y
        array_x[:,i,j]=x
        array_y[:,i,j]=y
    
        gaps_last[i,j] = max(np.dot(np.transpose(x),A))-min(np.dot(A,y))
        gaps_average[i,j] = max(np.dot(np.transpose(sum_x[:,j])/(i+1),A))-min(np.dot(A,sum_y[:,j]/(i+1)))
        
for j,alg in enumerate(bsp_array):
    plt.loglog(np.linspace(0,T-1, num=T), gaps_last[:,j], label = alg.get_name()+", last")
    plt.loglog(np.linspace(0,T-1, num=T), gaps_average[:,j], label = alg.get_name())
    print('Averaged x=',sum_x[:,j]/T, ', averaged y=',sum_y[:,j]/T)    
plt.xlabel('Rounds')
plt.ylabel('Duality Gap')
plt.legend()
plt.show()

for j,alg in enumerate(bsp_array):
    plt.plot(array_x[0,:,j],array_y[0,:,j], label = alg.get_name()+", last")
plt.xlabel('x_1')
plt.ylabel('y_1')
plt.legend()
plt.show()
