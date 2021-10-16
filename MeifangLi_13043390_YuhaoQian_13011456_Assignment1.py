import numpy as np
import matplotlib.pyplot as plt

def mandelbrot_test(c:complex,iter_max):
    # return: the number of iterations (<= maximum iteration)
    z=complex(0,0)
    iter_num=0
    for i in range(iter_max):
        z=z*z+c
        if np.abs(z) <=2:
            iter_num+=1
        else:
            break
    return iter_num

def area(x,y,i):
    count=0
    for j in range(int(len(x))):
        c = complex(x[j],y[j])
        iteration = mandelbrot_test(c, iter_max=i)
        if iteration == int(i):
            count += 1
        else:
            pass
        area = count/int(len(x)) * 6
    return area

def sample(n, type, x_low_limits, x_high_limits, y_low_limits, y_high_limits):
    X_list = []
    Y_list = []
    if type == "random":
        for i in range(n):
            x = np.random.uniform(x_low_limits, x_high_limits)
            y = np.random.uniform(y_low_limits, y_high_limits)
            X_list.append(x)
            Y_list.append(y)
        return X_list, Y_list

    if type == "Orthogonal":
        row_range = x_high_limits - x_low_limits
        col_range = y_high_limits - y_low_limits
        subsets = int(np.sqrt(n))
        row_count = subsets ** 2
        col_count = subsets ** 2
        x_subrange = row_range / row_count
        y_subrange = col_range / col_count
        # set initial condition: all rows and columns are not used as 0.
        rows = np.zeros(row_count)
        cols = np.zeros(col_count)
        for subset_x in range(subsets):
            for subset_y in range(subsets):
                # check every subset if options are used
                x_options = rows[subsets * subset_x:subsets * (subset_x + 1)] == 0
                # list all the free options index in the x_options
                x_free_options = [i + subset_x * subsets for i, value in enumerate(x_options) if value == True]
                # random choose one index of the free options
                x_chosen = np.random.choice(x_free_options, 1)
                # turn index into value
                x_value = np.random.uniform(low=x_low_limits + x_subrange * x_chosen,
                                            high=x_low_limits + x_subrange * (x_chosen + 1))
                # the chosen index of x change from 0 into 1
                rows[x_chosen] = 1
                X_list.append(x_value)

                # same with y
                y_options = cols[subsets * subset_y:subsets * (subset_y + 1)] == 0
                y_free_options = [i + subset_y * subsets for i, value in enumerate(y_options) if value == True]
                y_chosen = np.random.choice(y_free_options, 1)
                y_value = np.random.uniform(low=y_low_limits + y_subrange * y_chosen,
                                            high=y_low_limits + y_subrange * (y_chosen + 1))
                cols[y_chosen] = 1
                Y_list.append(y_value)

        return X_list, Y_list

    if type == "LHS":
        X_list = np.arange(0, n)
        X_list = x_low_limits + (X_list * (x_high_limits - x_low_limits) / n)
        X_list = X_list + (x_high_limits - x_low_limits) / n * np.squeeze(np.random.uniform(0, 1, (1, n)))

        Y_list = np.arange(0, n)
        np.random.shuffle(Y_list)
        Y_list = y_low_limits + (Y_list * (y_high_limits - y_low_limits) / n)
        Y_list = Y_list + (y_high_limits - y_low_limits) / n * np.squeeze(np.random.uniform(0, 1, (1, n)))

        return X_list, Y_list

#random sampling plot
test1=sample(4, type="random",  x_low_limits=-2, x_high_limits=2, y_low_limits=-2, y_high_limits=2)
X_random=test1[0]
Y_random=test1[1]

plt.figure(figsize=[4,4])
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.xlabel("pure random sampling")
plt.scatter(X_random,Y_random, c='r')
plt.show()

# orthogonal sampling plot
test2 = sample(4, type="Orthogonal", x_low_limits=-2, x_high_limits=2, y_low_limits=-2, y_high_limits=2)
X_ortho=test2[0]
Y_ortho=test2[1]

plt.figure(figsize=[4,4])
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.xlabel("Orthogonal sampling")
plt.scatter(X_ortho,Y_ortho, c='r')
plt.hlines(0,-2,2,linewidth=3,color="blue")
plt.vlines(0,-2,2,linewidth=3,color="blue")

for i in np.arange(-2,2,1):
    plt.axvline(i)
for i in np.arange(-2,2,1):
    plt.axhline(i)

plt.show()

# LHS sampling plot
test3 = sample(4, type="LHS", x_low_limits=-2, x_high_limits=2, y_low_limits=-2, y_high_limits=2)
X_lhs=test3[0]
Y_lhs=test3[1]

plt.figure(figsize=[4,4])
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.xlabel("Latin hypercube sampling")
plt.scatter(X_lhs,Y_lhs, c='r')

for i in np.arange(-2,2,1):
    plt.axvline(i)
for i in np.arange(-2,2,1):
    plt.axhline(i)

plt.show()

# generate 1200*800 pixels
x=np.linspace(-2,1,1200,endpoint=True)
y=np.linspace(-1,1,800,endpoint=True)
atlas = np.zeros((len(x), len(y))) #initial value of each pixel is 0

# assign value to each pixel according to iteration number
for i in range(len(x)):
    for j in range(len(y)):
        real=x[i]
        image=y[j]
        c=complex(real,image)
        atlas[i,j]=mandelbrot_test(c,iter_max=1000)

# color the pixels according to the iteration number
plt.imshow(atlas.T, interpolation='bilinear',extent=[-2,1,-1,1])
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.savefig('mandelbrot set itr1000',dpi=400)
plt.savefig('mandelbrot set itr50',dpi=400)
plt.show()

# fix s=1000, investigate the convergence with different iteration number
average_area=[]
for i in np.linspace(20,1000,50):
    area_list=[]
    for k in range(100):
        count = 0
        area = 0
        for s in range(1000):
            x = np.random.uniform(-2, 1)
            y = np.random.uniform(-1, 1)
            c = complex(x, y)
            iteration = mandelbrot_test(c, iter_max=int(i))
            if iteration == int(i):
                count += 1
            else:
                pass
            area = count / 1000 * 6
        area_list.append(area)
    average_area.append(np.average(area_list))
plt.plot(np.linspace(20,1000,50),np.abs(np.array(average_area)-np.array(average_area[-1])))
plt.xlabel('Iteration number, i')
plt.ylabel('Convergence rate')
plt.show()


space=np.linspace(10,100,19)
xais=[s**2 for s in space]

# random sampling, fix i=600, investigate the mean area and confidence interval with different s
random_average_list=[]
random_std_list=[]
random_CI_list=[]
for s in xais:
    area_list = []
    for simulation in range(100):
        random_set=sample(int(s), type="random",x_low_limits=-2,x_high_limits=1,y_low_limits=-1,y_high_limits=1)
        random_x=np.array(random_set[0])
        random_y=np.array(random_set[1])
        area_simu=area(random_x,random_y,600)
        area_list.append(area_simu)
    random_average_list.append(np.mean(area_list))
    random_std_list.append(np.std(area_list,ddof=1))
    random_CI_list.append(1.96*np.std(area_list,ddof=1)/np.sqrt(int(s)))

print(random_average_list)
print(random_std_list)
print(random_CI_list)
plt.plot(xais,random_average_list,label='Mean area')
plt.fill_between(xais,np.array(random_average_list)+np.array(random_CI_list),
                 np.array(random_average_list)-np.array(random_CI_list),
                 color='#FF4500',alpha=0.1, edgecolor="white",label='1.96*Std/sqrt(s)')
plt.xlabel('Sample size,s')
plt.ylabel('Area')
plt.legend(loc="upper right")
plt.show()

# LHS sampling, fix i=600, investigate the mean area and confidence interval with different s
lhs_average_list=[]
lhs_std_list=[]
lhs_CI_list=[]
for s in xais:
    area_list = []
    for simulation in range(100):
        lhs_set=sample(int(s), type="LHS",x_low_limits=-2,x_high_limits=1,y_low_limits=-1,y_high_limits=1)
        lhs_x=lhs_set[0]
        lhs_y=lhs_set[1]
        area_simu=area(lhs_x,lhs_y,600)
        area_list.append(area_simu)
    lhs_average_list.append(np.mean(area_list))
    lhs_std_list.append(np.std(area_list,ddof=1))
    lhs_CI_list.append(1.96*np.std(area_list,ddof=1)/np.sqrt(int(s)))

print(lhs_average_list)
print(lhs_std_list)
print(lhs_CI_list)
plt.plot(xais,lhs_average_list,label='Mean area')
plt.fill_between(xais,np.array(lhs_average_list)+np.array(lhs_CI_list),
                 np.array(lhs_average_list)-np.array(lhs_CI_list),
                 color='#FF4500',alpha=0.1, edgecolor="white",label='1.96*Std/sqrt(s)')
plt.xlabel('Sample size,s')
plt.ylabel('Area')
plt.legend(loc="upper right")
plt.savefig("LHS result",dpi=400)
plt.show()

# orthogonal sampling, fix i=600, investigate the mean area and confidence interval with different s
ortho_average_list=[]
ortho_std_list=[]
ortho_CI_list=[]
for s in xais:
    area_list = []
    for simulation in range(100):
        ortho_set=sample(int(s), type="Orthogonal",x_low_limits=-2,x_high_limits=1,y_low_limits=-1,y_high_limits=1)
        ortho_x=np.array(ortho_set[0])
        ortho_y=np.array(ortho_set[1])
        area_simu=area(ortho_x,ortho_y,600)
        area_list.append(area_simu)
    ortho_average_list.append(np.mean(area_list))
    ortho_std_list.append(np.std(area_list, ddof=1))
    ortho_CI_list.append(1.96*np.std(area_list, ddof=1)/np.sqrt(int(s)))

print(ortho_average_list)
print(ortho_std_list)
print(ortho_CI_list)
plt.plot(xais,ortho_average_list,label='Mean area')
plt.fill_between(xais,np.array(ortho_average_list)+np.array(ortho_CI_list),
                 np.array(ortho_average_list)-np.array(ortho_CI_list),
                 color='#FF4500',alpha=0.1, edgecolor="white",label='1.96*Std/sqrt(s)')
plt.xlabel('Sample size,s')
plt.ylabel('Area')
plt.legend(loc="upper right")
plt.savefig('Orthogonal')
plt.show()

#antithetic variables,using random sampling,fix i=600, investigate the mean area and confidence interval with different s
anti_average_list=[]
anti_std_list=[]
anti_CI_list=[]
for s in xais:
    area_list = []
    for simulation in range(100):
        anti_set=sample(int(s), type="random",x_low_limits=-2,x_high_limits=1,y_low_limits=-1,y_high_limits=1)
        anti_x1=anti_set[0]
        anti_y1 = anti_set[1]
        area_sim1 = area(anti_x1, anti_y1, 600)
        anti_x2 = [-1-i for i in anti_x1]
        anti_y2 = [-i for i in anti_y1]
        area_sim2 = area(anti_x2, anti_y2, 600)
        mean_area= (area_sim1 + area_sim2)/2
        area_list.append(mean_area)
    anti_average_list.append(np.mean(area_list))
    anti_std_list.append(np.std(area_list,ddof=1))
    anti_CI_list.append(1.96*np.std(area_list,ddof=1)/np.sqrt(int(s)))

print(anti_average_list)
print(anti_std_list)
print(anti_CI_list)
plt.plot(xais,anti_average_list,label='Mean area')
plt.fill_between(xais,np.array(anti_average_list)+np.array(anti_CI_list),
                 np.array(anti_average_list)-np.array(anti_CI_list),
                 color='#FF4500',alpha=0.1, edgecolor="white",label='1.96*Std/sqrt(s)')
plt.xlabel('Sample size,s')
plt.ylabel('Area')
plt.legend(loc="upper right")
plt.show()

