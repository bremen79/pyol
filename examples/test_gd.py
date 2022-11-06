import matplotlib.pyplot as plt
import numpy as np
import pyol
import math

d=3

A = np.random.rand(d,d)

coordinate_wise_kt = []
for i in range(d):
    bl = pyol.KT(d)
    coordinate_wise_kt.append(bl)

# AdaGrad, FTRL version
coordinate_wise_osd = []
for i in range(d):
    bl = pyol.L2OptimisticFTRL(d, lr_handler=pyol.LearningRateSqrtSumSquareGrads())
    coordinate_wise_osd.append(bl)

experts_array = [
                 pyol.L2OptimisticFTRL(d),
                 pyol.L2OptimisticOMD(d),
                 pyol.L2OptimisticFTRL(d, lr_handler=pyol.LearningRateSqrtSumSquareGrads()),
                 pyol.L2OptimisticOMD(d, lr_handler=pyol.LearningRateSqrtSumSquareGrads()),
                 pyol.KT(d),
                 pyol.CoordinateWise(d, coordinate_wise_kt),
                 pyol.CoordinateWise(d, coordinate_wise_osd)
             ]

num_alg = len(experts_array)

T=10000
avg_cumulative_loss = np.zeros([T+1,num_alg])

u=(2*np.random.rand(d)-1)*10

for i in range(T):
    for j,alg in enumerate(experts_array):
        
        # generate couples (x,y)
        z=2*np.random.rand(d)-1
        y=np.dot(z,u)+(2*np.random.rand(1)-1)*0.1
        
        x=alg.get_x()
        loss=abs(np.dot(x,z)-y)
        grad=np.sign(np.dot(x,z)-y)*z        
        alg.update(grad)
        avg_cumulative_loss[i+1,j] = (i*avg_cumulative_loss[i,j] + loss)/(i+1)


for j,alg in enumerate(experts_array):
    plt.loglog(np.linspace(0,T, num=T+1), avg_cumulative_loss[:,j], label = alg.get_name())
    print('Last x=',alg.get_x())

plt.xlabel('Rounds')
plt.ylabel('Average cumulative loss')
plt.legend()
plt.show()
