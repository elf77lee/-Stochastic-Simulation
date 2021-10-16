import random
import simpy
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.stats import bernoulli , shapiro
import seaborn as sns


def MMn_queuing(NEW_CUSTOMERS,INTERVAL_CUSTOMERS,SERVER):
    MMn_wait = []

    def source(env, number, interval, counter):
        """Source generates customers randomly"""
        for i in range(number):
            c = customer(env, 'Customer%02d' % i, counter, service_time=8.0)
            env.process(c)
            t = random.expovariate(1.0/ interval)
            yield env.timeout(t)


    def customer(env, name, counter, service_time):
        """Customer arrives, is served and leaves."""
        arrive = env.now

        with counter.request() as req:
            # Wait for the counter
            yield req

            wait = env.now - arrive
            MMn_wait.append(wait)

            ser_t = random.expovariate(1.0/ service_time)
            yield env.timeout(ser_t)

    # Setup and start the simulation
    env = simpy.Environment()

    # Start processes and run
    counter = simpy.Resource(env, capacity=SERVER)
    env.process(source(env, NEW_CUSTOMERS, INTERVAL_CUSTOMERS, counter))
    env.run()

    return MMn_wait

def MMn_priority_queuing(NEW_CUSTOMERS,INTERVAL_CUSTOMERS,SERVER):
    MMn_pri_wait = []

    def source(env, number, interval, counter):
        """Source generates customers randomly"""
        for i in range(number):
            c = customer(env, 'Customer%02d' % i, counter, service_time=8.0)
            env.process(c)
            t = random.expovariate(1.0 / interval)
            yield env.timeout(t)


    def customer(env, name, counter, service_time):
        """Customer arrives, is served and leaves."""
        arrive = env.now
        ser_t = random.expovariate(1.0 / service_time)

        with counter.request(priority=ser_t) as req:
            # Wait for the counter
            yield req

            wait = env.now - arrive
            MMn_pri_wait.append(wait)

            yield env.timeout(ser_t)

    # Setup and start the simulation
    env = simpy.Environment()

    # Start processes and run
    counter = simpy.PriorityResource(env, capacity=SERVER)
    env.process(source(env, NEW_CUSTOMERS, INTERVAL_CUSTOMERS, counter))
    env.run()

    return MMn_pri_wait

def MDn_queuing(NEW_CUSTOMERS,INTERVAL_CUSTOMERS,SERVER):
    MDn_wait = []

    def source(env, number, interval, counter):
        """Source generates customers randomly"""
        for i in range(number):
            c = customer(env, 'Customer%02d' % i, counter, service_time=8.0)
            env.process(c)
            t = random.expovariate(1.0 / interval)
            yield env.timeout(t)


    def customer(env, name, counter, service_time):
        """Customer arrives, is served and leaves."""
        arrive = env.now

        with counter.request() as req:
            # Wait for the counter
            yield req

            wait = env.now - arrive
            MDn_wait.append(wait)

            ser_t = service_time
            yield env.timeout(ser_t)

    # Setup and start the simulation
    env = simpy.Environment()

    # Start processes and run
    counter = simpy.Resource(env, capacity=SERVER)
    env.process(source(env, NEW_CUSTOMERS, INTERVAL_CUSTOMERS, counter))
    env.run()

    return MDn_wait

def MLTn_queuing(NEW_CUSTOMERS,INTERVAL_CUSTOMERS,SERVER):
    MLTn_wait = []

    def source(env, number, interval, counter):
        """Source generates customers randomly"""
        for i in range(number):
            c = customer(env, 'Customer%02d' % i, counter)
            env.process(c)
            t = random.expovariate(1.0/ interval)
            yield env.timeout(t)


    def customer(env, name, counter):
        """Customer arrives, is served and leaves."""
        arrive = env.now

        bernoulli = scipy.stats.bernoulli.rvs(0.75,size=1)
        if bernoulli == 1:
            ser_t = random.expovariate(1.0/ 1)
        else:
            ser_t = random.expovariate(1.0/ 5)

        with counter.request() as req:
            # Wait for the counter
            yield req

            wait = env.now - arrive
            MLTn_wait.append(wait)

            yield env.timeout(ser_t)

    # Setup and start the simulation
    env = simpy.Environment()

    # Start processes and run
    counter = simpy.Resource(env, capacity=SERVER)
    env.process(source(env, NEW_CUSTOMERS, INTERVAL_CUSTOMERS, counter))
    env.run()

    return MLTn_wait

def MMn(customer,interval,counter,servicetime):
    wait_list = []
    def arrive_customer(env, number, interval, counter):
        for i in range(number):
            c = get_service(env, counter, average_service_time=servicetime)
            env.process(c)
            t = random.expovariate(1/interval)
            yield env.timeout(t)

    def get_service(env, counter, average_service_time):

        arrive_time = env.now

        with counter.request() as req:
            yield req
            wait = env.now - arrive_time
            wait_list.append(wait)

            service_time = random.expovariate( 1/average_service_time)
            yield env.timeout(service_time)
    env = simpy.Environment()
    counter = simpy.Resource(env, capacity=counter)
    env.process(arrive_customer(env,customer,interval,counter))
    env.run()
    #return wait_list[int(0.2*len(wait_list)):int(0.8*len(wait_list))]
    return wait_list

def MMn2(customer,interval,counter,servicetime):
    wait_list = []
    def arrive_customer(env, number, interval, counter):
        for i in range(number):
            c = get_service(env, counter, average_service_time=servicetime)
            env.process(c)
            t = random.expovariate(1/interval)
            yield env.timeout(t)

    def get_service(env, counter, average_service_time):

        arrive_time = env.now

        with counter.request() as req:
            yield req
            wait = env.now - arrive_time
            wait_list.append(wait)

            service_time = random.expovariate( 1/average_service_time)
            yield env.timeout(service_time)
    env = simpy.Environment()
    counter = simpy.Resource(env, capacity=counter)
    env.process(arrive_customer(env,customer,interval,counter))
    env.run()
    return wait_list[int(0.2*len(wait_list)):]
    #return wait_list

"""
non-mathematical comparison of mean waiting time about MM1 and MM2
"""
rho_list = np.linspace(0.1, 0.99, 100)
mu = 1 / 8
MM1_list = []
MM2_list = []

for i in rho_list:
    MM1 = i / (mu * (1 - i))  # E(W)=rho/(mu*(1-rho))
    MM2 = i ** 2 / (mu * (1 - i ** 2))  # E(W)=rho**2/(mu(1-rho**2))
    MM1_list.append(MM1)
    MM2_list.append(MM2)

plt.plot(rho_list, MM1_list, label='MM1')
plt.plot(rho_list, MM2_list, label='MM2')
plt.xlabel("$\\rho$")
plt.ylabel("Mean waiting time")
plt.legend(loc="best")
plt.savefig("theory comparison", dpi=400)
plt.show()

'''Different number of customers with fixed rho
'''
#MM4 rho=0.8
simulation=100
customer_number=[100,1000,5000,10000,15000,20000]
interval_ave=2.5
server_num=4
service_ave=8

for number in customer_number:
    waiting = []
    for simu in range(simulation):
        waiting.append(np.mean(MMn(customer=number,
                                   interval=interval_ave,
                                   counter=server_num,
                                   servicetime=service_ave)))
    sns.distplot(waiting,bins=10,label='customer%s'%(number))
    print('AverageWaitingTime:',np.mean(waiting))
rho = service_ave / (server_num * interval_ave)
print('rho', rho)
plt.xlabel('Mean waiting time')
plt.ylabel('Probability density')
plt.legend()
plt.xlim(-2,15)
plt.savefig('customersrho08',dpi=400)
plt.show()

MM4 rho=0.4
simulation=100
customer_number=[100,500,1000,2000,3000,4000]
interval_ave=5
server_num=4
service_ave=8

for number in customer_number:
    waiting = []
    for simu in range(simulation):
        waiting.append(np.mean(MMn(customer=number,
                                   interval=interval_ave,
                                   counter=server_num,
                                   servicetime=service_ave)))
    sns.distplot(waiting,bins=10,label='customer%s'%(number))
    print('AverageWaitingTime:',np.mean(waiting))
rho = service_ave / (server_num * interval_ave)
print('rho', rho)
plt.xlabel('Mean waiting time')
plt.ylabel('Probability density')
plt.legend()
plt.xlim(-0.5,1.5)
#plt.savefig('customersrho04',dpi=400)
plt.show()
#
'''increase customer until pass normality test'''
def normalityTest(startnumber,interval,counter,servicetime):
    number = startnumber
    count = 0
    while number < 100000:
        waiting = []
        for simu in range(100):
            waiting.append(np.mean(MMn(customer=number,
                                       interval=interval,
                                       counter=counter,
                                       servicetime=servicetime)))
        normality = stats.shapiro(waiting).pvalue
        print('')
        print('Mean', np.mean(waiting))
        if normality > 0.05:
            print('Pass Normality test', number)
            count += 1
            number += 1000
            if count > 5:
                print('---------stop-----------')
                break
        else:
            print(number, ':', normality)
            number += 1000
print('---MM1--rho=0.1------------------------')
normalityTest(1000,interval=80,counter=1,servicetime=8) #rho=0.1
print('---MM1--rho=0.4------------------------')
normalityTest(1000,interval=20,counter=1,servicetime=8) #rho=0.4
print('---MM1--rho=0.8------------------------')
normalityTest(1000,interval=10,counter=1,servicetime=8) #rho=0.8

print('---MM4--rho=0.1------------------------')
normalityTest(1000,interval=20,counter=4,servicetime=8) #rho=0.1
print('---MM4--rho=0.4------------------------')
normalityTest(1000,interval=5,counter=4,servicetime=8) #rho=0.4
print('---MM4--rho=0.8------------------------')
normalityTest(1000,interval=2.5,counter=4,servicetime=8) #rho=0.8

'''comparison with cut the first 20% data'''
simulation=100
customer_number=10000
interval_ave=2.5
server_num=4
service_ave=8   #rho=0.8
waiting=[]
waitingcut=[]
for simu in range(simulation):
    waiting.append(np.mean(MMn(customer=customer_number,
                               interval=interval_ave,
                               counter=server_num,
                               servicetime=service_ave)))
    # in MMn2(), the first 20% data is cut, remove the warming up period
    waitingcut.append(np.mean(MMn2(customer=customer_number,
                               interval=interval_ave,
                               counter=server_num,
                               servicetime=service_ave)))
sns.distplot(waiting,bins=10,label='Cut first 20%')
sns.distplot(waitingcut,bins=10,label='Full data')
print('Cut 20% ',np.mean(waitingcut),'pvalue',stats.shapiro(waitingcut))
print('Full',np.mean(waiting),'pvalue',stats.shapiro(waiting))
plt.xlabel('Mean waiting time')
plt.ylabel('Probability density')
plt.legend()
plt.savefig('cuteffect',dpi=400)
plt.show()


''' check MM4 priority steady state'''
simulation=100
customer_number=[100,1000,5000,10000,15000,20000]
interval_ave=2.5
server_num=4
service_ave=8

for number in customer_number:
    waiting = []
    for simu in range(simulation):
        waiting.append(np.mean(MMn_priority_queuing(number,interval_ave,server_num)))
    sns.distplot(waiting,bins=10,label='customer%s'%(number))
    print('AverageWaitingTime:',np.mean(waiting))
rho = service_ave / (server_num * interval_ave)
print('rho', rho)
plt.xlabel('Mean waiting time')
plt.ylabel('Probability density')
plt.legend()
plt.xlim(0,6)
plt.savefig('customersrho08prior',dpi=400)
plt.show()

''' check M/D/4 steady state'''
simulation=100
customer_number=[100,1000,5000,10000,15000,20000]
interval_ave=2.5
server_num=4
service_ave=8

for number in customer_number:
    waiting = []
    for simu in range(simulation):
        waiting.append(np.mean(MDn_queuing(number,interval_ave,server_num)))
    sns.distplot(waiting,bins=10,label='customer%s'%(number))
    print('AverageWaitingTime:',np.mean(waiting))
rho = service_ave / (server_num * interval_ave)
print('rho', rho)
plt.xlabel('Mean waiting time')
plt.ylabel('Probability density')
plt.legend()
plt.xlim(0,6)
plt.savefig('customersrho08MD4',dpi=400)
plt.show()
''' check M/LT/4 steady state'''
simulation=100
customer_number=[100,1000,5000,10000,15000,20000]
interval_ave=2.5/4
server_num=4
service_ave=2

for number in customer_number:
    waiting = []
    for simu in range(simulation):
        waiting.append(np.mean(MLTn_queuing(number,interval_ave,server_num)))
    sns.distplot(waiting,bins=10,label='customer%s'%(number))
    print('AverageWaitingTime:',np.mean(waiting))
rho = service_ave / (server_num * interval_ave)
print('rho', rho)
plt.xlabel('Mean waiting time')
plt.ylabel('Probability density')
plt.legend()
plt.xlim(0,6)
plt.savefig('customersrho08longtail',dpi=400)
plt.show()

'''Normality test for M/M/4 priority,M/D/4,M/LT/4 at rho=0.8,customer=10000'''

waiting = []
for simu in range(100):
        waiting.append(np.mean(MMn_queuing(10000,2.5,4)))
print('M/M/4',shapiro(waiting))
waiting = []
for simu in range(100):
        waiting.append(np.mean(MDn_queuing(10000,2.5,4)))
print('M/D/4',shapiro(waiting))
waiting = []
for simu in range(100):
        waiting.append(np.mean(MLTn_queuing(10000,2.5/4,4)))
print('M/LT/4',shapiro(waiting))

'''Long tail distribution'''
b_set=[]
for i in range(2000):
    bernoulli = scipy.stats.bernoulli.rvs(0.75, size=1)
    if bernoulli == 1:
        ser_t = random.expovariate(1/ 1)
    else:
        ser_t = random.expovariate(1/ 5)
    b_set.append(ser_t)
print('mean',np.mean(b_set))
plt.vlines(2,0,0.5,label='Mean=2',colors='red')
sns.distplot(b_set,bins=20,label='Long-tail')
plt.legend()
plt.xlabel('value')
plt.xlim(-5,20)
plt.savefig('longtail',dpi=400)
plt.show()



"""
MMn queuing with different rho
"""
interval_1=[160,80,80/1.5,40,80/2.5,80/3,80/3.5,20,80/4.5,16,80/5.5,80/6,80/6.5,80/7,80/7.5,10,80/8.5,80/9,80/9.5]
interval_2=[i/2 for i in interval_1]
interval_4=[i/4 for i in interval_1]

rho_1=[8/i for i in interval_1]
MM1_wait_avg_list=[]
MM1_wait_std_list=[]
MM1_wait_ci_list=[]
for i in interval_1:
    MM1_wait_avg=[]
    for j in range(100):
        MM1_wait=MMn_queuing(10000,i,1)
        MM1_wait_avg.append(np.mean(MM1_wait))
    MM1_wait_avg_list.append(np.mean(MM1_wait_avg))
    MM1_wait_std_list.append(np.std(MM1_wait_avg,ddof=1))
    MM1_wait_ci_list.append(1.96*np.std(MM1_wait_avg,ddof=1)/np.sqrt(100))
print(MM1_wait_avg_list)
print(MM1_wait_std_list)
print(MM1_wait_ci_list)

rho_2=[4/i for i in interval_2]
MM2_wait_avg_list=[]
MM2_wait_std_list=[]
MM2_wait_ci_list=[]
for i in interval_2:
    MM2_wait_avg=[]
    for j in range(100):
        MM2_wait=MMn_queuing(10000,i,2)
        MM2_wait_avg.append(np.mean(MM2_wait))
    MM2_wait_avg_list.append(np.mean(MM2_wait_avg))
    MM2_wait_std_list.append(np.std(MM2_wait_avg,ddof=1))
    MM2_wait_ci_list.append(1.96*np.std(MM2_wait_avg,ddof=1)/np.sqrt(100))
print(MM2_wait_avg_list)
print(MM2_wait_std_list)
print(MM2_wait_ci_list)

rho_4=[2/i for i in interval_4]
MM4_wait_avg_list=[]
MM4_wait_std_list=[]
MM4_wait_ci_list=[]
for i in interval_4:
    MM4_wait_avg=[]
    for j in range(100):
        MM4_wait=MMn_queuing(10000,i,4)
        MM4_wait_avg.append(np.mean(MM4_wait))
    MM4_wait_avg_list.append(np.mean(MM4_wait_avg))
    MM4_wait_std_list.append(np.std(MM4_wait_avg,ddof=1))
    MM4_wait_ci_list.append(1.96*np.std(MM4_wait_avg,ddof=1)/np.sqrt(100))
print(MM4_wait_avg_list)
print(MM4_wait_std_list)
print(MM4_wait_ci_list)

plt.plot(rho_1,MM1_wait_avg_list,color="red",label="Mean:n=1")
plt.plot(rho_2,MM2_wait_avg_list,color="orange",label="Mean:n=2")
plt.plot(rho_4,MM4_wait_avg_list,color="green",label="Mean:n=4")
plt.fill_between(rho_1,np.array(MM1_wait_avg_list)+np.array(MM1_wait_ci_list),
                 np.array(MM1_wait_avg_list)-np.array(MM1_wait_ci_list),
                 color="lightsalmon",alpha=0.1,label='confidence interval:n=1')
plt.fill_between(rho_2,np.array(MM2_wait_avg_list)+np.array(MM2_wait_ci_list),
                 np.array(MM2_wait_avg_list)-np.array(MM2_wait_ci_list),
                 color="bisque",alpha=0.1,label='confidence interval:n=2')
plt.fill_between(rho_4,np.array(MM4_wait_avg_list)+np.array(MM4_wait_ci_list),
                 np.array(MM4_wait_avg_list)-np.array(MM4_wait_ci_list),
                 color="lightgreen",alpha=0.1,label='confidence interval:n=4')
plt.xlabel("$\\rho$")
plt.ylabel("Mean waiting time")
plt.yscale("log")
plt.legend(loc="best")
plt.savefig("MMn mean")
plt.show()


"""
MMn priority queuing with different rho
"""
rho_1=[8/i for i in interval_1]
MM1_pri_wait_avg_list=[]
MM1_pri_wait_std_list=[]
MM1_pri_wait_ci_list=[]
for i in interval_1:
    MM1_pri_wait_avg=[]
    for j in range(100):
        MM1_pri_wait=MMn_priority_queuing(10000,i,1)
        MM1_pri_wait_avg.append(np.mean(MM1_pri_wait))
    MM1_pri_wait_avg_list.append(np.mean(MM1_pri_wait_avg))
    MM1_pri_wait_std_list.append(np.std(MM1_pri_wait_avg,ddof=1))
    MM1_pri_wait_ci_list.append(1.96*np.std(MM1_pri_wait_avg,ddof=1)/np.sqrt(100))
print(MM1_pri_wait_avg_list)
print(MM1_pri_wait_std_list)
print(MM1_pri_wait_ci_list)

rho_2=[4/i for i in interval_2]
MM2_pri_wait_avg_list=[]
MM2_pri_wait_std_list=[]
MM2_pri_wait_ci_list=[]
for i in interval_2:
    MM2_pri_wait_avg=[]
    for j in range(100):
        MM2_pri_wait=MMn_priority_queuing(10000,i,2)
        MM2_pri_wait_avg.append(np.mean(MM2_pri_wait))
    MM2_pri_wait_avg_list.append(np.mean(MM2_pri_wait_avg))
    MM2_pri_wait_std_list.append(np.std(MM2_pri_wait_avg,ddof=1))
    MM2_pri_wait_ci_list.append(1.96*np.std(MM2_pri_wait_avg,ddof=1)/np.sqrt(100))
print(MM2_pri_wait_avg_list)
print(MM2_pri_wait_std_list)
print(MM2_pri_wait_ci_list)

rho_4=[2/i for i in interval_4]
MM4_pri_wait_avg_list=[]
MM4_pri_wait_std_list=[]
MM4_pri_wait_ci_list=[]
for i in interval_4:
    MM4_pri_wait_avg=[]
    for j in range(100):
        MM4_pri_wait=MMn_priority_queuing(10000,i,4)
        MM4_pri_wait_avg.append(np.mean(MM4_pri_wait))
    MM4_pri_wait_avg_list.append(np.mean(MM4_pri_wait_avg))
    MM4_pri_wait_std_list.append(np.std(MM4_pri_wait_avg,ddof=1))
    MM4_pri_wait_ci_list.append(1.96*np.std(MM4_pri_wait_avg,ddof=1)/np.sqrt(100))
print(MM4_pri_wait_avg_list)
print(MM4_pri_wait_std_list)
print(MM4_pri_wait_ci_list)

plt.plot(rho_1,MM1_pri_wait_avg_list,color="red",label="Mean:n=1")
plt.plot(rho_2,MM2_pri_wait_avg_list,color="orange",label="Mean:n=2")
plt.plot(rho_4,MM4_pri_wait_avg_list,color="green",label="Mean:n=4")
plt.fill_between(rho_1,np.array(MM1_pri_wait_avg_list)+np.array(MM1_pri_wait_ci_list),
                 np.array(MM1_pri_wait_avg_list)-np.array(MM1_pri_wait_ci_list),
                 color="lightsalmon",alpha=0.1,label='confidence interval:n=1')
plt.fill_between(rho_2,np.array(MM2_pri_wait_avg_list)+np.array(MM2_pri_wait_ci_list),
                 np.array(MM2_pri_wait_avg_list)-np.array(MM2_pri_wait_ci_list),
                 color="bisque",alpha=0.1,label='confidence interval:n=2')
plt.fill_between(rho_4,np.array(MM4_pri_wait_avg_list)+np.array(MM4_pri_wait_ci_list),
                 np.array(MM4_pri_wait_avg_list)-np.array(MM4_pri_wait_ci_list),
                 color="lightgreen",alpha=0.1,label='confidence interval:n=4')
plt.xlabel("$\\rho$")
plt.ylabel("Mean waiting time")
plt.yscale("log")
plt.legend(loc="best")
plt.savefig("MMn mean priority",dpi=400)
plt.show()

"""MM1 compare to MM1-priority"""
plt.plot(rho_1,MM1_wait_avg_list,color="red",label="M/M/1")
plt.plot(rho_1,MM1_pri_wait_avg_list,color="green",label="M/M/1-priority")
plt.fill_between(rho_1,np.array(MM1_wait_avg_list)+np.array(MM1_wait_ci_list),
                 np.array(MM1_wait_avg_list)-np.array(MM1_wait_ci_list),
                 color="red",alpha=0.1,label='confidence interval')
plt.fill_between(rho_1,np.array(MM1_pri_wait_avg_list)+np.array(MM1_pri_wait_ci_list),
                 np.array(MM1_pri_wait_avg_list)-np.array(MM1_pri_wait_ci_list),
                 color="red",alpha=0.1,label='confidence interval')
plt.xlabel("$\\rho$")
plt.ylabel("Mean waiting time")
plt.yscale("log")
plt.legend(loc="best")
plt.savefig("MM1 with priority comparison",dpi=400)
plt.show()

"""
MDn queuing with different rho
"""
rho_1=[8/i for i in interval_1]
MD1_wait_avg_list=[]
MD1_wait_std_list=[]
MD1_wait_ci_list=[]
for i in interval_1:
    MD1_wait_avg=[]
    for j in range(100):
        MD1_wait=MDn_queuing(10000,i,1)
        MD1_wait_avg.append(np.mean(MD1_wait))
    MD1_wait_avg_list.append(np.mean(MD1_wait_avg))
    MD1_wait_std_list.append(np.std(MD1_wait_avg,ddof=1))
    MD1_wait_ci_list.append(1.96*np.std(MD1_wait_avg,ddof=1)/np.sqrt(100))
print(MD1_wait_avg_list)
print(MD1_wait_std_list)
print(MD1_wait_ci_list)

rho_2=[4/i for i in interval_2]
MD2_wait_avg_list=[]
MD2_wait_std_list=[]
MD2_wait_ci_list=[]
for i in interval_2:
    MD2_wait_avg=[]
    for j in range(100):
        MD2_wait=MDn_queuing(10000,i,2)
        MD2_wait_avg.append(np.mean(MD2_wait))
    MD2_wait_avg_list.append(np.mean(MD2_wait_avg))
    MD2_wait_std_list.append(np.std(MD2_wait_avg,ddof=1))
    MD2_wait_ci_list.append(1.96*np.std(MD2_wait_avg,ddof=1)/np.sqrt(100))
print(MD2_wait_avg_list)
print(MD2_wait_std_list)
print(MD2_wait_ci_list)

rho_4=[2/i for i in interval_4]
MD4_wait_avg_list=[]
MD4_wait_std_list=[]
MD4_wait_ci_list=[]
for i in interval_4:
    MD4_wait_avg=[]
    for j in range(100):
        MD4_wait=MDn_queuing(10000,i,4)
        MD4_wait_avg.append(np.mean(MD4_wait))
    MD4_wait_avg_list.append(np.mean(MD4_wait_avg))
    MD4_wait_std_list.append(np.std(MD4_wait_avg,ddof=1))
    MD4_wait_ci_list.append(1.96*np.std(MD4_wait_avg,ddof=1)/np.sqrt(100))
print(MD4_wait_avg_list)
print(MD4_wait_std_list)
print(MD4_wait_ci_list)

plt.plot(rho_1,MD1_wait_avg_list,color="red",label="Mean:n=1")
plt.plot(rho_2,MD2_wait_avg_list,color="orange",label="Mean:n=2")
plt.plot(rho_4,MD4_wait_avg_list,color="green",label="Mean:n=4")
plt.fill_between(rho_1,np.array(MD1_wait_avg_list)+np.array(MD1_wait_ci_list),
                 np.array(MD1_wait_avg_list)-np.array(MD1_wait_ci_list),
                 color="lightsalmon",alpha=0.1,label='confidence interval:n=1')
plt.fill_between(rho_2,np.array(MD2_wait_avg_list)+np.array(MD2_wait_ci_list),
                 np.array(MD2_wait_avg_list)-np.array(MD2_wait_ci_list),
                 color="bisque",alpha=0.1,label='confidence interval:n=2')
plt.fill_between(rho_4,np.array(MD4_wait_avg_list)+np.array(MD4_wait_ci_list),
                 np.array(MD4_wait_avg_list)-np.array(MD4_wait_ci_list),
                 color="lightgreen",alpha=0.1,label='confidence interval:n=4')
plt.xlabel("$\\rho$")
plt.ylabel("Mean waiting time")
plt.yscale("log")
plt.legend(loc="best")
plt.savefig("MDn mean",dpi=400)
plt.show()

"""
MLTn queuing with different rho
"""
interval=[160,80,80/1.5,40,80/2.5,80/3,80/3.5,20,80/4.5,16,80/5.5,80/6,80/6.5,80/7,80/7.5,10,80/8.5,80/9,80/9.5]
interval_1=[i/4 for i in interval]
interval_2=[i/2 for i in interval_1]
interval_4=[i/4 for i in interval_1]

rho_1=[2/i for i in interval_1]
MLT1_wait_avg_list=[]
MLT1_wait_std_list=[]
MLT1_wait_ci_list=[]
for i in interval_1:
    MLT1_wait_avg=[]
    for j in range(100):
        MLT1_wait=MLTn_queuing(10000,i,1)
        MLT1_wait_avg.append(np.mean(MLT1_wait))
    MLT1_wait_avg_list.append(np.mean(MLT1_wait_avg))
    MLT1_wait_std_list.append(np.std(MLT1_wait_avg,ddof=1))
    MLT1_wait_ci_list.append(1.96*np.std(MLT1_wait_avg,ddof=1)/np.sqrt(100))
print(MLT1_wait_avg_list)
print(MLT1_wait_std_list)
print(MLT1_wait_ci_list)

rho_2=[1/i for i in interval_2]
MLT2_wait_avg_list=[]
MLT2_wait_std_list=[]
MLT2_wait_ci_list=[]
for i in interval_2:
    MLT2_wait_avg=[]
    for j in range(100):
        MLT2_wait=MLTn_queuing(10000,i,2)
        MLT2_wait_avg.append(np.mean(MLT2_wait))
    MLT2_wait_avg_list.append(np.mean(MLT2_wait_avg))
    MLT2_wait_std_list.append(np.std(MLT2_wait_avg,ddof=1))
    MLT2_wait_ci_list.append(1.96*np.std(MLT2_wait_avg,ddof=1)/np.sqrt(100))
print(MLT2_wait_avg_list)
print(MLT2_wait_std_list)
print(MLT2_wait_ci_list)

rho_4=[0.5/i for i in interval_4]
MLT4_wait_avg_list=[]
MLT4_wait_std_list=[]
MLT4_wait_ci_list=[]
for i in interval_4:
    MLT4_wait_avg=[]
    for j in range(100):
        MLT4_wait=MLTn_queuing(10000,i,4)
        MLT4_wait_avg.append(np.mean(MLT4_wait))
    MLT4_wait_avg_list.append(np.mean(MLT4_wait_avg))
    MLT4_wait_std_list.append(np.std(MLT4_wait_avg,ddof=1))
    MLT4_wait_ci_list.append(1.96*np.std(MLT4_wait_avg,ddof=1)/np.sqrt(100))
print(MLT4_wait_avg_list)
print(MLT4_wait_std_list)
print(MLT4_wait_ci_list)

plt.plot(rho_1,MLT1_wait_avg_list,color="red",label="Mean:n=1")
plt.plot(rho_2,MLT2_wait_avg_list,color="orange",label="Mean:n=2")
plt.plot(rho_4,MLT4_wait_avg_list,color="green",label="Mean:n=4")
plt.fill_between(rho_1,np.array(MLT1_wait_avg_list)+np.array(MLT1_wait_ci_list),
                 np.array(MLT1_wait_avg_list)-np.array(MLT1_wait_ci_list),
                 color="lightsalmon",alpha=0.1,label='confidence interval:n=1')
plt.fill_between(rho_2,np.array(MLT2_wait_avg_list)+np.array(MLT2_wait_ci_list),
                 np.array(MLT2_wait_avg_list)-np.array(MLT2_wait_ci_list),
                 color="bisque",alpha=0.1,label='confidence interval:n=2')
plt.fill_between(rho_4,np.array(MLT4_wait_avg_list)+np.array(MLT4_wait_ci_list),
                 np.array(MLT4_wait_avg_list)-np.array(MLT4_wait_ci_list),
                 color="lightgreen",alpha=0.1,label='confidence interval:n=4')
plt.xlabel("$\\rho$")
plt.ylabel("Mean waiting time")
plt.yscale("log")
plt.legend(loc="best")
plt.savefig("MLTn mean",dpi=400)
plt.show()


"""
rho=0.8, comparison of mean waiting time between different methods
"""
MMn=[MM1_wait_avg_list[-4],MM2_wait_avg_list[-4],MM4_wait_avg_list[-4]]
MMn_up=[MM1_wait_avg_list[-4]+MM1_wait_ci_list[-4],
        MM2_wait_avg_list[-4]+MM2_wait_ci_list[-4],
        MM4_wait_avg_list[-4]+MM4_wait_ci_list[-4]]
MMn_down=[MM1_wait_avg_list[-4]-MM1_wait_ci_list[-4],
        MM2_wait_avg_list[-4]-MM2_wait_ci_list[-4],
        MM4_wait_avg_list[-4]-MM4_wait_ci_list[-4]]


MMn_pri=[MM1_pri_wait_avg_list[-4],MM2_pri_wait_avg_list[-4],MM4_pri_wait_avg_list[-4]]
MMn_pri_up=[MM1_pri_wait_avg_list[-4]+MM1_pri_wait_ci_list[-4],
            MM2_pri_wait_avg_list[-4]+MM2_pri_wait_ci_list[-4],
            MM4_pri_wait_avg_list[-4]+MM4_pri_wait_ci_list[-4]]
MMn_pri_down=[MM1_pri_wait_avg_list[-4]-MM1_pri_wait_ci_list[-4],
            MM2_pri_wait_avg_list[-4]-MM2_pri_wait_ci_list[-4],
            MM4_pri_wait_avg_list[-4]-MM4_pri_wait_ci_list[-4]]

MDn=[MD1_wait_avg_list[-4],MD2_wait_avg_list[-4],MD4_wait_avg_list[-4]]
MDn_up=[MD1_wait_avg_list[-4]+MD1_wait_ci_list[-4],
        MD2_wait_avg_list[-4]+MD2_wait_ci_list[-4],
        MD4_wait_avg_list[-4]+MD4_wait_ci_list[-4]]
MDn_down=[MD1_wait_avg_list[-4]-MD1_wait_ci_list[-4],
        MD2_wait_avg_list[-4]-MD2_wait_ci_list[-4],
        MD4_wait_avg_list[-4]-MD4_wait_ci_list[-4]]

MLTn=[MLT1_wait_avg_list[-4],MLT2_wait_avg_list[-4],MLT4_wait_avg_list[-4]]
MLTn_up=[MLT1_wait_avg_list[-4]+MLT1_wait_ci_list[-4],
        MLT2_wait_avg_list[-4]+MLT2_wait_ci_list[-4],
        MLT4_wait_avg_list[-4]+MLT4_wait_ci_list[-4]]
MLTn_down=[MLT1_wait_avg_list[-4]-MLT1_wait_ci_list[-4],
        MLT2_wait_avg_list[-4]-MLT2_wait_ci_list[-4],
        MLT4_wait_avg_list[-4]-MLT4_wait_ci_list[-4]]

server=[1,2,4]
plt.plot(server,MMn,'o-',color="red",label="M/M/n")
plt.plot(server,MMn_pri,'o-',color="orange",label="M/M/n-priority")
plt.plot(server,MDn,'o-',color="green",label="M/D/n")
plt.plot(server,MLTn,'o-',color="blue",label="M/LT/n")
plt.fill_between(server,MMn_up,MMn_down,
                 color="red",alpha=0.1,label='confidence interval')
plt.fill_between(server,MMn_pri_up,MMn_pri_down,
                 color="orange",alpha=0.1,label='confidence interval')
plt.fill_between(server,MDn_up,MDn_down,
                 color="green",alpha=0.1,label='confidence interval')
plt.fill_between(server,MLTn_up,MLTn_down,
                 color="blue",alpha=0.1,label='confidence interval')
plt.xlabel("server")
plt.ylabel("Mean waiting time")
plt.xticks([1,2,4])
plt.legend(loc="best")
plt.savefig("compare methods",dpi=400)
plt.show()

