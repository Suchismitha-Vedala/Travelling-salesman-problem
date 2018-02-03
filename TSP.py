import csv
import numpy as np
import pandas as pd
import sys, os,copy
import math,time
import  random
start=time.time()

best=True
ng=8000
pop_size=25
tsize=20
m=0.01


def calculate_tsp1(x,y):
    if x==y:
        cost=0
    elif (x<3 and y<3):
        cost=1
    elif x<3:
        cost=200
    elif y<3:
        cost=200
    elif (x%7==y%7):
        cost=2
    else:
        cost=abs(x-y)+3
    return cost
def calculate_tsp2(x,y):
    if (x==y):
        cost=0
    elif (x+y<10):
        cost = abs(x-y)+4
    elif ((x+y)%11==0):
        cost=3
    else:
        cost=(abs(x-y)**2)+10
    return cost
def calculate_tsp3(x,y):
    if (x==y):
        cost = 0 
    else :
        cost = (x+y)**2
    return cost
'''
def Genetic_Algorithm(ng,pop_size):
        print "Creates the population:"
        population = RoutePop(pop_size, True)
        print "Finished Creation of the population"
        initial_length = population.fittest.length
        best_route = Route()

        for x in range(ng):       
            new_population = GA().evolve_population(population)

            # If we have found a new shorter route, save it to best_route
            if new_population.fittest.length < best_route.length:
                best_route = (new_population.fittest)
            # Prints info to the terminal:
            #print 'generation number:' , x
            
            print 'Current Length' , (best_route.length)
    
        #print('Finished evolving {0} generations.'.format(n_generations))
       # print("Elapsed time was {0:.1f} seconds.".format(end_time - start_time))
       # print(' ')
        #print('Initial best distance: {0:.2f}'.format(initial_length))
        #print('Final best distance:   {0:.2f}'.format(best_route.length))
        print('The best route went via:')
        best_route.print_cities()
        
        return best_route



class Route(object):
    
    def __init__(self):
        random.shuffle(l)
        self.route =sorted(l, key=lambda *args: random.random())
        self.calculate_length()

    def calculate_length(self):
        
       
        self.length = 0.0
        for i in self.route:
            end=len(self.route)
            next_city = self.route[self.route.index(i)+1-end]
            dist_to_next = distance_matrix[i][next_city]
            self.length = self.length+ dist_to_next

    def print_cities(self):
        a=[]
        for city in self.route:
            a.append(city)
        
        print a

    def is_valid_route(self):
        if (len(self.route) != len(set(self.route))):
            return False
        return True

    
class RoutePop(object):
   
    def __init__(self, size, initialise):
        self.rt_pop = []
        self.size = size
        if initialise:
            for x in range(0,size):
                new_rt = Route()
                self.rt_pop.append(new_rt)
            self.get_fittest()

    def get_fittest(self):
        # sorts the list based on the routes' lengths
        sorted_list = sorted(self.rt_pop, key=lambda x: x.length, reverse=False)
        self.fittest = sorted_list[0]
        return self.fittest

class GA(object):

    def crossover(self, parent1, parent2):
     
        child = Route()
        for x in range(0,len(child.route)):
            child.route[x] = None
        r=len(parent1.route)
        
        
        p1 = random.randint(0,r)
        p2 = random.randint(0,r)
        if(p1>p2):
            p1,p2=p2,p1
            
        for i in range(p1,p2):
            child.route[i] = parent1.route[i] 
       
        for i in range(len(parent2.route)):
            if  parent2.route[i] not in child.route:
                for j in range(len(child.route)):
                    if child.route[j] == None:
                        child.route[j] = parent2.route[i]
                        break
        # repeated until all the cities are in the child route
        # returns the child route (of type Route())
        child.calculate_length()
        return child

    def mutate(self, route1):
        
        if random.random() < m:
            # two random indices:
            m1 = random.randint(0,len(route1.route)-1)
            m2 = random.randint(0,len(route1.route)-1)
            # if they're the same, skip to the chase
            if (m1 == m2):
                return route1
            # Otherwise swap them:
            city1 = route1.route[m1]
            city2 = route1.route[m2]
            route1.route[m2] = city1
            route1.route[m1] = city2
        # Recalculate the length of the route (updates it's .length)
        route1.calculate_length()

        return route1

    
    
    def tournament_select(self, population):
     
        # New smaller population (not intialised)
        tournament_pop = RoutePop(size=tsize,initialise=False)
        #sorted_list = sorted(population.rt_pop, key=lambda x: x.length, reverse=False)
        for i in range(tsize):
            tournament_pop.rt_pop.append(random.choice(population.rt_pop))                                         
        return tournament_pop.get_fittest()

    def evolve_population(self, population):
      
        #makes a new population:
        new_population = RoutePop(size=population.size, initialise=True)
        # Elitism offset (amount of Routes() carried over to new population)
        #elitismOffset = 0
        # if we have elitism, set the first of the new population to the fittest of the old
       
        new_population.rt_pop[0] = population.fittest
            #elitismOffset = 1
        # Goes through the new population and fills it with the child of two tournament winners from the previous populatio
        for x in range(1,new_population.size):
            # two parents:
            parent1 = self.tournament_select(population)
            parent2 = self.tournament_select(population)
            child = self.crossover(parent1,parent2)
            new_population.rt_pop[x] = child

        # Mutates all the routes (mutation with happen with a prob p = k_mut_prob)
        for route in new_population.rt_pop:
            if random.random() < m:
                self.mutate(route)

        # Update the fittest route:
        new_population.get_fittest()

        return new_population

'''
def simplestrategy(N,c,meb):
    l=[i for i in range(N)]
    df=pd.DataFrame(index=l,columns=l)
    if(c == "c1"):
        for i in l:
            for j in l:
                df[i][j]=calculate_tsp1(i,j)
    if(c=="c2"):
        for i in l:
            for j in l:
                df[i][j]=calculate_tsp2(i,j)
    if(c=="c3"):
        for i in l:
            for j in l:
                df[i][j]=calculate_tsp3(i,j)
    print df
    routes=[]
    global count
    count=0
    for i in range(N):
        visited=[]
        city=list(xrange(N))
        distance=0
        dist=[]
        temp=[]
        c=city.pop(i)
        visited.append(c)
        while(len(city)>0):
            temp=[]
            dist=[]
            for j in city:
                if(j!=c):
                    temp.append([df.loc[c][j],j])
                    
            #print temp
            for i in range(len(temp)):
                dist.append(temp[i][0])
            #print dist
            lowest_cost=min(dist)
           
            index=dist.index(lowest_cost)
            #print index
            distance=distance+lowest_cost
            distance
            c=temp[index][1]
            count+=1
            del city[index]
            #print city
            visited.append(c)
        
        visited.append(visited[0])
        distance+=df.loc[visited[-2]][visited[-1]]
        routes.append([visited,distance])
        
        
        
    short=[]
    for i in range(len(routes)):
        short.append(routes[i][1])
    
    ij=short.index(min(short))  
    print routes[ij][0],routes[ij][1]
    with open("output_simple.txt","w") as f:
        f.write('Number of cities :  \n')
        f.write('%d' % N )
        f.write('\n')
        f.write('optimal distance :')
        f.write('%d' % min(short) )
        f.write('\n optimal path \n')
        f.write('\n')
        for i in routes[ij][0]:
            f.write('%d\t' % i)
        f.write('\n')
        f.write('meb value\n')
        f.write('%d' % count)
     



    '''
    output_file="./output"+"_simple_"+str(time.time())+".csv"
    with open(output_file) as fw:
        wr = csv.writer(fw, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        wr.writerow("simple strategy for Travelling salesman Problem")
        wr.write('\n')
        wr.writerow(routes[ij][0])
        wr.write('\n')
        wr.write( routes[ij][1])
        wr.write('\n')
        wr.close()
    '''
    #return routes[ij][0],routes[ij][1]

def probability(cnew, cold, T):
    d=cnew-cold
    if d <= 0:
        p = 1
    elif d > 0:
        p = math.exp((cold-cnew) / T)
    return float(p)
 
def swap_cities(path):
    length = len(path) - 1
    a = random.randint(0, length)
    b = random.randint(0, length)
    if a is not b:
        path[a],path[b]=path[b],path[a]
    else:
        path[a],path[a - 1]=path[a-1],path[a]
    return path
 
def  calculate_length(path,df1):
    cost=0
    length=len(path)
    for i in range(length-1):
        cost=cost+df1[i][i+1]
    cost=cost+df1[path[-1]][path[0]]
    return cost
 
def sa(T1,T2,beta,N,meb,df):
    T = T1
    Tmin = T2
    B = beta
    # Create a list of cities
    route1 = [i for i in range(N)]
 
    random.shuffle(route1)
    cost1 = calculate_length(route1,df)
    count = 0
    k = 0
 
    best = [cost1, route1]
    while (T > Tmin and count < meb):
 
        route2 = swap_cities(route1)
        cost2 = calculate_length(route2,df)

 
 
        if(cost2 < best[0]):
            best[0] = cost2
            best[1] = route2
         
    
        
        elif random.random() <= probability(cost2,cost1, random.random() * T):
            T = T* B
            route1 = route2
            cost1 = cost2
            k =k+ 1
        count=count+1
        print best[0],count,T
 
    print("========================================================================================")
    print("optimal_route is :", str(best[1]))
    print("optimal_distance is :", str(best[0]))
    with open("output_sophisticated.txt","w") as f:
        f.write('Number of cities \n')
        f.write('%d' % N )
        f.write('\n')
        f.write('optimal distance \n')
        f.write('\n')
        f.write('%d' % best[0])
        f.write('\n optimal path \n')
        f.write('\n')
        for i in best[1]:
            f.write('%d\t' % i)
        f.write('\n')
        f.write('meb value\n')
        f.write('%d' % count)
     

       
def sophstrategy(N,c,meb):
    T1=12000#input('Enter initial temperature')
    T2=0.1#input('enter cooling temperature')
    b=0.998
    meb=meb
    N=N
    l=[i for i in range(N)]
    df=pd.DataFrame(index=l,columns=l)
    if(c == "c1"):
        for i in l:
            for j in l:
                df[i][j]=calculate_tsp1(i,j)
    if(c=="c2"):
        for i in l:
            for j in l:
                df[i][j]=calculate_tsp2(i,j)
    if(c=="c3"):
        for i in l:
            for j in l:
                df[i][j]=calculate_tsp3(i,j)
    sa(T1,T2,b,N,meb,df)

    
def main(start):
    #choice = input('Enter the method of input: 1. Terminal 2. Csv File')
    #if(choice==1):
    response = input("Please enter a number for the chosen search strategy : 1.Simple 2.Sophisticated.  ")
    meb=input("Please enter a maximum effort bound.  ")
    n=input("Please enter number of cities.  ")
    c= raw_input("Please enter cost function: c1 or c2 or c3.  " )
      # response = raw_input("Please enter a number for the chosen search strategy : 1.Simple 2.Sophisticated")     
    print n,c,meb
    if (int(response)==1):
        simplestrategy(n,c,meb)
    if (response==2):
        sophstrategy(n,c,meb)
    #if(choice==2):
     #   file=raw_input("Please enter csv file name and path.  " )



    end=time.time()
    t=end-start
    print 'Time elapsed' , t


    






if __name__ == '__main__':
    start=time.time()
    main(start)