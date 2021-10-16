import numpy as np
import matplotlib.pyplot as plt
import timeit

class Coordinate(object):
    def __init__(self, data):
        self.data = data

    def get_distance(self, a, b):
        return np.sqrt((a[1]-b[1])**2+(a[2]-b[2])**2)

    def get_total_distance(self, coords):
        dist = 0
        for first, second in zip(coords[:-1], coords[1:]):
            dist += self.get_distance(first, second)
        dist += self.get_distance(coords[0], coords[-1])
        return dist

    def change_state(self, data):
        r1, r2 = np.random.randint(1, len(data)+1, size=2)
        data[min(r1, r2):max(r1, r2)] = data[min(r1, r2):max(r1, r2)][::-1]
        return data

    def simu_annealing_exp(self, data, type, alpha, T_init, iter_num, MCL):
        self.type = type
        self.T_init = T_init
        self.iter_num = iter_num
        self.MCL = MCL
        self.alpha = alpha
        if type == "Exponential":
            cost_list = []
            sol = data.copy()
            cost_list.append(self.get_total_distance(data))
            for i in range(iter_num): #set outer loop number
                T = T_init*np.power(alpha, i) #exponential
                for j in range(int(MCL)):#markov chain length-inner loop
                    pre_data = sol.copy()
                    cost0 = self.get_total_distance(sol)
                    new_data = self.change_state(sol)

                    # get the new cost
                    cost1 = self.get_total_distance(new_data)

                    if cost1 < cost0:
                        # accept the new solution
                        sol = new_data
                    else:
                        # accept the new(worse) solution with a given probability
                        x = np.random.uniform()
                        if x < np.exp((cost0 - cost1) / T):
                            # accept the solution
                            sol = new_data
                        else:
                            # do not accept the solution
                            sol = pre_data
                cost_list.append(self.get_total_distance(sol))
            return cost_list


    def simu_annealing(self, data, type, T_init, iter_num, MCL):
        self.type = type
        self.T_init = T_init
        self.iter_num = iter_num
        self.MCL = MCL
        if type == "Linear":
            costlist = []
            sol = data.copy()
            costlist.append(self.get_total_distance(data))  #
            for i in range(iter_num):  # set outer loop number
                # print(i, 'th iteration', 'cost=', cost0)
                T = T_init - 0.3333 * i  # linear
                for j in range(int(MCL)):  # markov chain length-inner loop
                    pre_data = sol.copy()
                    cost0 = self.get_total_distance(sol)
                    new_data = self.change_state(sol)  # new state
                    cost1 = self.get_total_distance(new_data)  # get the new cost
                    if cost1 < cost0:  # accept the new solution
                        sol = new_data
                    else:
                        x = np.random.uniform()
                        if x < np.exp((cost0 - cost1) / T):
                            sol = new_data  # accept the solution
                        else:
                            # do not accept the solution
                            sol = pre_data
                costlist.append(self.get_total_distance(sol))
            return sol, costlist

        if type == "Exponential":
            costlist=[]
            sol = data.copy()
            costlist.append(self.get_total_distance(data)) #
            for i in range(iter_num):  # set outer loop number
                #print(i, 'th iteration', 'cost=', cost0)
                T = T_init * np.power(0.8, i)  # exponential
                for j in range(int(MCL)):  # markov chain length-inner loop
                    pre_data = sol.copy()
                    cost0 = self.get_total_distance(sol)
                    new_data = self.change_state(sol)  # new state
                    cost1 = self.get_total_distance(new_data)  # get the new cost
                    if cost1 < cost0:  # accept the new solution
                        sol = new_data
                    else:
                        x = np.random.uniform()
                        if x < np.exp((cost0 - cost1) / T):
                            sol = new_data  # accept the solution
                        else:
                            # do not accept the solution
                            sol = pre_data
                costlist.append(self.get_total_distance(sol))
            return sol,costlist

        if type == "Quadratic":
            costlist = []
            sol = data.copy()
            costlist.append(self.get_total_distance(data))  #
            for i in range(iter_num):  # set outer loop number
                # print(i, 'th iteration', 'cost=', cost0)
                T = T_init / (1 + 53.54 * (i ** 2))  # quadratic
                for j in range(int(MCL)):  # markov chain length-inner loop

                    pre_data = sol.copy()

                    cost0 = self.get_total_distance(sol)
                    new_data = self.change_state(sol)  # new state
                    cost1 = self.get_total_distance(new_data)  # get the new cost
                    if cost1 < cost0:  # accept the new solution
                        sol = new_data
                    else:
                        x = np.random.uniform()
                        if x < np.exp((cost0 - cost1) / T):
                            sol = new_data  # accept the solution
                        else:
                            # do not accept the solution
                            sol = pre_data
                costlist.append(self.get_total_distance(sol))
            return sol, costlist

# data = np.loadtxt('/Users/yuhao/Downloads/Lecture/Stochastic Simulation/Ass3/TSP-Configurations/eil51.tspcut.txt', dtype=int)
# eil51 = Coordinate(data)

'''Different initial Temperature 30 simulation,process'''
###280, expoential, alpha=0.8, MCL=100, outeriteration=300
data280=np.loadtxt('/Users/yuhao/Downloads/Lecture/Stochastic Simulation/Ass3/TSP-Configurations/a280.tspcopy.txt', dtype=int)
data280[1:60]=data280[1:60][::-1]  #cost 3258.636207486553
a280 = Coordinate(data280)

Temperature=[10,50,100,150,200]
x=np.arange(0,301,1)
simulation=30
for temp in Temperature:
    cost_list =[]
    for simu in range(simulation):
        solution,costlist = a280.simu_annealing(data280, type='Exponential', T_init=temp, iter_num=300, MCL=100)
        cost_list.append(costlist)

    mean_list = []
    std_list = []
    for j in range(301):
        list=[]
        for k in range(simulation):
            list.append(cost_list[k][j])
        mean_list.append(np.mean(list))
        std_list.append(np.std(list))
    plt.plot(x,mean_list,label='T= %d'%(temp))
    plt.fill_between(x,np.array(mean_list)-np.array(std_list)
                 ,np.array(mean_list)+np.array(std_list)
                 ,alpha=0.3,label='std T= %d'%(temp))
    print('Processed', temp)
    print(temp,'meanlist', mean_list)
    print(temp,'std list', std_list)


plt.xlabel('Cooling steps')
plt.ylabel('Distance')
plt.legend(loc="best")
plt.savefig('expTiprocessWithStd',dpi=400)
plt.clf()

'''Different initial Temperature 30 simulation, result with std'''

Temperature=[10,50,100,150,200]
optima=[]
optimastd=[]
simulation=30
for temp in Temperature:
    solutionlist=[]
    for simu in range(simulation):
        solution, costlist = a280.simu_annealing(data280, type="Exponential", T_init=temp, iter_num=300, MCL=100)
        solutionlist.append(costlist[-1])
    mean=np.mean(solutionlist)
    std=np.std(solutionlist,ddof=1)
    print('Temp:',temp,'Mean:',mean,'std:',std)
    plt.vlines(temp,mean-std,mean+std,color='palevioletred',linestyle='--')
    optima.append(mean)
    optimastd.append(std)
print('Mean',optima)
print('std',optimastd)

plt.fill_between(Temperature,
                 np.array(optima)-np.array(optimastd)
                 ,np.array(optima)+np.array(optimastd)
                 ,color='plum',alpha=0.3,label='std')
plt.plot(Temperature,optima,'o-',label='Mean Optima')
plt.xlabel('Initial Temperature')
plt.ylabel('Optima')
plt.legend()
plt.savefig('exp30simulationResult',dpi=400)
plt.clf()

'''best optima'''
#a280 = Coordinate(data280)
solution, costlist = a280.simu_annealing(data280, type="Exponential", T_init=100, iter_num=1000, MCL=200)
plt.plot(costlist)
plt.xlabel('Cooling steps')
plt.ylabel('Distance')
plt.legend()
plt.savefig('bestoptimaprocess',dpi=400)
plt.show()

###30 simulation
opt=[]
for simu in range(30):
    solution, costlist = a280.simu_annealing(data280, type="Exponential", T_init=100, iter_num=1000, MCL=200)
    opt.append(costlist[-1])
    print('processed',simu)
print('optima mean',np.mean(opt),',std',np.std(opt,ddof=1))

### show route of 280 cities
solution, costlist = a280.simu_annealing(data280, type="Exponential", T_init=100, iter_num=1000, MCL=200)
print(costlist[-1])
x=[solution[i][1] for i in range(len(solution))]
y=[solution[i][2] for i in range(len(solution))]
plt.plot(x, y, 'o', color='orange',markersize=3)
plt.arrow(x[-1], y[-1], (x[0]-x[-1]), (y[0]-y[-1]),color ='blue', head_width=1.5,length_includes_head=True,)
for i in range(len(x)-1):
    plt.arrow(x[i], y[i], (x[i+1]-x[i]), (y[i+1]-y[i]),color ='blue', head_width=1.5, length_includes_head=True)
plt.savefig('bestoptima280',dpi=400)
plt.show()




'''The effect of outer iteration number, T steps  '''
outerstep=[150,300,600,1000,2000]
simulation=30
x = np.arange(0, 2001, 1)
cost_list =[]
for simu in range(simulation):
        solution,costlist = a280.simu_annealing(data280, type='Exponential', T_init=100, iter_num=2000, MCL=100)
        cost_list.append(costlist)
        print('processed',simu)
mean_list = []
std_list = []
for j in range(2001):
        list=[]
        for k in range(simulation):
            list.append(cost_list[k][j])
        mean_list.append(np.mean(list))
        std_list.append(np.std(list))
print('meanlist', mean_list)
print('std list', std_list)
print(mean_list[150],mean_list[300],mean_list[600],mean_list[1000],mean_list[2000])
print(std_list[150],std_list[300],std_list[600],std_list[1000],std_list[2000])

plt.plot(x,mean_list,label='Cooling step')
plt.fill_between(x,np.array(mean_list)-np.array(std_list)
                 ,np.array(mean_list)+np.array(std_list)
                 ,alpha=0.3,label='std')

plt.xlabel('Cooling steps')
plt.ylabel('Distance')
plt.legend(loc="best")
plt.savefig('expouterloop',dpi=400)
plt.show()
###plot result with std
mean=[4897.456717060316,4033.5314924397035,3430.553424192819,3133.6380919091343,2962.4787875031266]
std=[187.73809967225534,114.06543699642647,82.06287975805809,63.52832930042443,59.513672092608076]
for i in range(4):
    plt.vlines(outerstep[i],mean[i]-std[i],mean[i]+std[i],color='palevioletred',linestyle='--')
plt.fill_between(outerstep,
                 np.array(mean)-np.array(std)
                 ,np.array(mean)+np.array(std)
                 ,color='plum',alpha=0.3,label='std')
plt.plot(outerstep,mean,'o-',label='Mean Optima')
plt.xlabel('Cooling Steps')
plt.ylabel('Optima')
plt.legend()
plt.savefig('expSperateCoolingStep')
plt.show()

'''different cooling schedules example'''
T_init=100

T_linear = [T_init-0.3333*i for i in range(300)]
T_exponential=[T_init*np.power(0.97, i) for i in range(300)]
T_quadratic=[T_init/(1+53.54*(i**2)) for i in range(300)]

i=np.linspace(0,299,300)
plt.plot(i,T_linear, label="linear,alpha=0.3333")
plt.plot(i, T_exponential, label="exponential,alpha=0.97")
plt.plot(i, T_quadratic, label="quadratic,alpha=53.54")
plt.xlabel("step")
plt.ylabel("T")
plt.xlim(0,300)
plt.legend(loc="best")
plt.savefig('schedule sample', dpi=400)
plt.show()

'''comparison between differrent cooling schedules'''
###280, exp_alpha=0.97, lin_alpha=0.3333, qua_alpha=53.54, T_init=100, outloopmax=300, MCL=100
data280=np.loadtxt('/Users/elflee/Jupyter-notebook/280test.txt', dtype=int)
data280[1:60]= data280[1:60][::-1]
a280 = Coordinate(data280)
type=["Linear", "Exponential", "Quadratic"]
x=np.arange(0,301,1)

for i in type:
    cost_list =[]
    for simu in range(30):
        costlist = a280.simu_annealing(data280, type=i, T_init=100, iter_num=300, MCL=100)
        cost_list.append(costlist)
    mean_list = []
    std_list = []
    for j in range(301):
        list=[]
        for k in range(30):
            list.append(cost_list[k][j])
        mean_list.append(np.mean(list))
        std_list.append(np.std(list))
    print('Type', i, 'mean', mean_list)
    print('Type', i, 'std',std_list)
    plt.plot(x,mean_list,label=i)
    plt.fill_between(x,np.array(mean_list)-np.array(std_list)
                 ,np.array(mean_list)+np.array(std_list)
                 ,alpha=0.3,label='std')
plt.xlabel('Cooling steps')
plt.ylabel('Distance')
plt.legend(loc="best")
plt.savefig('cooling schedule',dpi=400)
plt.show()



'''Different Markov Chain Length '''
###280, exp, alpha=0.8, T_init=100, outloopmax=300
data280=np.loadtxt('/Users/elflee/Jupyter-notebook/280test.txt', dtype=int)
data280[1:60]= data280[1:60][::-1]
a280 = Coordinate(data280)
Markov_length=[50, 100, 150, 200, 250]
x=np.arange(0,301,1)

for i in Markov_length:
    cost_list =[]
    for simu in range(30):
        costlist = a280.simu_annealing(data280, type="Exponential", T_init=100, iter_num=300, MCL=i)
        cost_list.append(costlist)
    mean_list = []
    std_list = []
    for j in range(301):
        list=[]
        for k in range(30):
            list.append(cost_list[k][j])
        mean_list.append(np.mean(list))
        std_list.append(np.std(list))
    print('MCL', i, 'mean', mean_list)
    print('MCL', i, 'std',std_list)
    plt.plot(x,mean_list,label='MCL= %d'%(i))
    plt.fill_between(x,np.array(mean_list)-np.array(std_list)
                 ,np.array(mean_list)+np.array(std_list)
                 ,alpha=0.3,label='std MCL= %d'%(i))
plt.xlabel('Cooling steps')
plt.ylabel('Distance')
plt.legend(loc="best")
plt.savefig('different MCL',dpi=400)
plt.show()

###plot result with std
optima=[4236.847715428929, 4060.5797849056084, 3775.7897697993635, 3521.4016518548, 3373.4094932393514]
optima_std=[165.52683796173784, 107.47150922951016, 101.54785138586051, 79.20549078181678, 56.83354739855402]
plt.fill_between(Markov_length,
                 np.array(optima)-np.array(optima_std)
                 ,np.array(optima)+np.array(optima_std)
                 ,color='plum',alpha=0.3,label='std')
plt.plot(Markov_length,optima,'o-',label='Mean Optima')
plt.vlines(Markov_length,np.array(optima)-np.array(optima_std)
                 ,np.array(optima)+np.array(optima_std),color='palevioletred',linestyle='--')
plt.xlabel('Markov chain length')
plt.ylabel('Optima')
plt.legend()
plt.savefig('expMCL',dpi=400)
plt.show()

"""Different alpha for exponential"""
###280, exp,  T_init=100, outloopmax=300, MCL=100
data280=np.loadtxt('/Users/elflee/Jupyter-notebook/280test.txt', dtype=int)
data280[1:60]= data280[1:60][::-1]
a280 = Coordinate(data280)
alpha_list=[0.8,0.85,0.9,0.95,0.99]
x=np.arange(0,301,1)

for i in alpha_list:
    cost_list =[]
    for simu in range(30):
        costlist = a280.simu_annealing_exp(data280, type="Exponential", alpha=i, T_init=100, iter_num=300, MCL=100)
        cost_list.append(costlist)
    mean_list = []
    std_list = []
    for j in range(301):
        list=[]
        for k in range(30):
            list.append(cost_list[k][j])
        mean_list.append(np.mean(list))
        std_list.append(np.std(list))
    print('Alpha', i, 'mean', mean_list)
    print('Alpha', i, 'std',std_list)
    plt.plot(x,mean_list,label='alpha= %.2f'%(i))
    plt.fill_between(x,np.array(mean_list)-np.array(std_list)
                 ,np.array(mean_list)+np.array(std_list)
                 ,alpha=0.3,label='std alpha= %.2f'%(i))
plt.xlabel('Cooling steps')
plt.ylabel('Distance')
plt.legend(loc="best")
plt.savefig('different alpha',dpi=400)
plt.show()

###plot result with std
optima=[4047.875976933804, 4125.912344956227, 4315.395015095566, 4491.260362054431, 6365.195594190342]
optima_std=[138.5751294703274, 125.7684649869006, 106.0530224968542, 126.96833795455741, 165.01662135110521]
plt.fill_between(alpha_list,
                 np.array(optima)-np.array(optima_std)
                 ,np.array(optima)+np.array(optima_std)
                 ,color='plum',alpha=0.3,label='std')
plt.plot(alpha_list,optima,'o-',label='Mean Optima')
plt.vlines(alpha_list,np.array(optima)-np.array(optima_std)
                 ,np.array(optima)+np.array(optima_std),color='palevioletred',linestyle='--')
plt.xlabel('Alpha for exponential cooling')
plt.ylabel('Optima')
plt.legend()
plt.savefig('expAlpha',dpi=400)
plt.show()




'''test for pcb442'''
data442=np.loadtxt('/Users/yuhao/Downloads/Lecture/Stochastic Simulation/Ass3/TSP-Configurations/pcb442.tspcopy.txt', dtype='double')
pcb442=Coordinate(data442)
print('Cost0',pcb442.get_total_distance(data442))
start = timeit.default_timer()
solution442, costlist442 = pcb442.simu_annealing(data442, type="Exponential", T_init=100, iter_num=2000, MCL=250)
stop = timeit.default_timer()
print('Time',stop-start)
print('optima442',costlist442[-1])
x=[solution442[i][1] for i in range(len(solution442))]
y=[solution442[i][2] for i in range(len(solution442))]
plt.plot(x, y, 'o', color='orange',markersize=3)
plt.arrow(x[-1], y[-1], (x[0]-x[-1]), (y[0]-y[-1]),color ='blue', head_width=1.5,length_includes_head=True,)
for i in range(len(x)-1):
    plt.arrow(x[i], y[i], (x[i+1]-x[i]), (y[i+1]-y[i]),color ='blue', head_width=1.5, length_includes_head=True)
plt.savefig('bestoptima442',dpi=400)
plt.show()