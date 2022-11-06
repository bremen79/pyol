import matplotlib.pyplot as plt
import numpy as np
import pyol

d=3

experts_array = [pyol.EntropyOptimisticFTRL(d, hint_handler=pyol.HintPreviousGrad(d)),
             pyol.RM(d),
             pyol.RMPlus(d),
             pyol.KTExperts(d),
             pyol.AdaHedge(d),
             pyol.L2OptimisticFTRL(d, lr_handler=pyol.LearningRateSqrtSumSquareGrads(), proj_handler=pyol.ProjSimplex()),
             pyol.L2OptimisticOMD(d, lr_handler=pyol.LearningRateSqrtSumSquareGrads(), proj_handler=pyol.ProjSimplex())
             ]

num_alg = len(experts_array)

T=10000
avg_cumulative_loss = np.zeros([T+1,num_alg])

for i in range(T):
    for j,alg in enumerate(experts_array):
        
        # generate random loss of the experts
        grad=np.random.rand(d)
        grad[0] *= 0.9
        grad[1] *= 0.8
        grad[2] *= 0.7
        
        x=alg.get_x()
        alg.update(grad)
        loss=np.dot(x,grad)
        avg_cumulative_loss[i+1,j] = (i*avg_cumulative_loss[i,j] + loss)/(i+1)
        
for j,alg in enumerate(experts_array):
    plt.loglog(np.linspace(0,T, num=T+1), avg_cumulative_loss[:,j], label = alg.get_name())
    print('x=',alg.get_x())
plt.xlabel('Rounds')
plt.ylabel('Average cumulative loss')
plt.legend()
plt.show()
