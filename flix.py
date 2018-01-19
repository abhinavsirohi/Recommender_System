import datetime
import networkx as nx
import random
import numpy as np
import math
import itertools
G=nx.read_gml("karate.gml",label='id')
#G=pd.read_csv('./flixster/out.txt', delimiter=' ')
edge=G.edges()
node=G.nodes()
nodes = len(node)
adj_matrix= [[0]*nodes for i in range(nodes)]
edges= len(edge)#input("Number of Edges: ")
edge_final=[]
for i in edge:
    u=i[0]
    v=i[1]
    edge_final.append([u,v])
    adj_matrix[u-1][v-1]=adj_matrix[v-1][u-1]=1
print("your Adjacency Matrix is:\n")
for i in adj_matrix:
    print (i)

#Similarity_matrix
degree=[0]*nodes
for i in range(nodes):
    k=0
    for j in range(nodes):
        if (adj_matrix[i][j]==1):
            k=k+1
    degree[i]=k
Similarity_matrix= [[0]*nodes for i in range(nodes)]

for i in range(nodes):
        d=[0]*nodes
        for j in range(nodes):
                q=0.0
                if(adj_matrix[i][j]==1):
                        for p in range(nodes):
                                if(adj_matrix[i][p]==1 & adj_matrix[j][p]==1):
                                        q=q+(1.0/degree[p])

                d[j]=q
        Similarity_matrix[i]=d
print ("your Similarity Matrix is:\n")
for i in Similarity_matrix:
    print (i)
    
    
#MSNS
x=[]
for j in range(nodes):
	z=0.0
	for i in range(nodes):
		if(Similarity_matrix[j][i]>z):
			z=Similarity_matrix[j][i]
	x.append(z)
MSN=[[]for i in range(nodes)]
for j in range(nodes):
	for i in range(nodes):
		if(Similarity_matrix[j][i]==x[j]):
			MSN[j].append(i)
print ("MSN is:")
temp=[[] for i in MSN]
for i in range(len(MSN)):
	temp[i]=[x+1 for x in MSN[i]]
print (temp)	

#qwert
e=[]
a=-1
n=[]
for i in range(nodes):
	n.append("gray")
	
for i in range(len(MSN)):
	if(n[i]=="gray"):
		for j in range(len(MSN[i])):
			w=MSN[i][j]
			if(n[w] == "gray"):
				p=1
			else:
				p=0
				break
		if(p==1):
			a=a+1
			e.append([])
		e[a].append(i)
		n[i]="black"
		for j in range(len(MSN[i])):
			k=MSN[i][j]
			if(n[k]=="gray"):
				e[a].append(k)
				n[k]="black"
print ("Your Community list is:")

t=[[] for i in e]
for i in range(len(e)):
	t[i]=[x+1 for x in e[i]]
print (t)	
#-----------------------------------------------------------------------------#


#Influence calculation
def influence(seed):
    global G
    iterations=10000           
                    # Number of Iterations
    seed_set=seed   # Selecting intial seed set randomly
    print('Selected Seeds:',seed)
    avg_influence=0.0
    for i in range(iterations):            
        S=seed_set
        nx.set_edge_attributes(G,(1.0-random.random()),'edge_prob')
        nx.set_node_attributes(G,(1.0-random.random()),'node_prob')
        
        for i in range(len(S)):                 # Process each node in seed set
            for neighbor in list(G[S[i]]):    
                if G[S[i]][neighbor]['edge_prob']>G.node[neighbor]['node_prob']:           # Generate a random number and compare it with propagation probability
                    if neighbor not in S:       
                        S.append(neighbor)
        avg_influence += (float(len(S))/iterations)
    print(avg_influence)
    return(int(round(avg_influence)))
#-----------------------------------------------------------------------------#


#algo.pdf
t1=datetime.datetime.now()
n=nodes
beta_g=0
budget=500
nx.set_node_attributes(G,random.randint(1,500),'cost')
for i in node:
    beta_g+=G.node[i]['cost']

beta_k=list(np.zeros((len(e)),dtype=np.int))
f_k=list(np.zeros((len(e)),dtype=np.int))
n_k=list(np.zeros((len(e)),dtype=np.int))
cut=np.zeros((len(e),len(e)),dtype=np.int)
for i in range (0,len(e)):
    nk_i=len(t[i])
    n_k[i]=nk_i/n
    for j in t[i]:
        beta_k[i]=beta_k[i]+G.node[j]['cost']
    
    f_k[i]=beta_k[i]/beta_g
    for j in range (i+1,len(e)):
        for k in t[i]:
            neighbors=list(G[k])
            for com_ele in t[j]:
                if com_ele in neighbors:
                    cut[i][j]=cut[i][j]+1


S=[]     
beta_ka=list(np.zeros((len(e)),dtype=np.int))
for i in range(0,len(e)):
    D=[]
    beta_ka[i]=budget*(f_k[i]+n_k[i])/2
    for u in t[i]:
        l=0
        for v in t[i]:
            if v in list(G[u]) and v!=u:
                l=l+1
        D.append(l)        
                
    D_index=sorted(range(len(D)), key=lambda k: D[k],reverse=True)
    S_i=[]
    count=0
    while beta_ka[i]>0:            #ask this, it might not be so exact as it seems to be
        for k in D_index:
            count+=1
            if G.node[t[i][k]]['cost']<=beta_ka[i]:
                S.append(t[i][k])
                beta_ka[i]=beta_ka[i]-G.node[t[i][k]]['cost']
        
        j=np.argmax(cut[i])
        beta_ka[j]=beta_ka[j]+beta_ka[i]
        beta_ka[i]=0
print("Influence: ",influence(S))
print("Time1: ",datetime.datetime.now()-t1)
#-----------------------------------------------------------------------------#



#Random 
t2=datetime.datetime.now()
random_nodes = list(G.nodes())
random_budget = budget
random_seed = []
while random_budget > 0 and len(random_nodes) > 0:
    a = random.choice(random_nodes)
    count = 0
    if G.node[a]['cost'] <= random_budget:
        if count == 1:
            while not random_seed[-1]:
                random_seed.pop()
        random_seed.append(a)
        random_nodes.remove(a)
        random_budget -= G.node[a]['cost']
    else:
        count=1
        random_nodes.remove(a)
        random_seed.append(0)
        if random_seed[-26:-1] == [0]*25:
            while not random_seed[-1]:
                random_seed.pop()
            break
print("Influence: ",influence(random_seed))
print("Time2: ",datetime.datetime.now()-t2)
#-----------------------------------------------------------------------------#



#Maximum degree heuristic
t3=datetime.datetime.now()
max_nodes=list(G.nodes())
deg=[]
for i in max_nodes:
    deg.append(len(list(G[i])))
max_nodes_index=sorted(range(len(deg)), key = lambda k:deg[k],reverse=True)
print(max_nodes_index)
max_budget = budget
max_seed = []
while max_budget > 0:
    for i in max_nodes_index:
        if G.node[max_nodes[i]]['cost'] <= max_budget:
            max_seed.append(max_nodes[i])
            max_budget -= G.node[max_nodes[i]]['cost']
    max_budget = 0
print("Influence: ",influence(max_seed))
print("Time3: ",datetime.datetime.now()-t3)
#-----------------------------------------------------------------------------#


#Maximum clustering coeff. heuristic
t4=datetime.datetime.now()
clustering_nodes=list(G.nodes())
clustering_coeff=list(nx.clustering(G).values())
clustering_coeff_index=sorted(range(len(clustering_coeff)), key = lambda k:clustering_coeff[k],reverse=True)
print(max_nodes_index)
clustering_budget = budget
clustering_seed = []
while clustering_budget > 0:
    for i in clustering_coeff_index:
        if G.node[clustering_nodes[i]]['cost'] <= clustering_budget:
            clustering_seed.append(clustering_nodes[i])
            clustering_budget -= G.node[clustering_nodes[i]]['cost']
    clustering_budget = 0
print("Influence: ",influence(clustering_seed))
print("Time4: ",datetime.datetime.now()-t4)
#-----------------------------------------------------------------------------#


#MIOA building
def MIOA(graph,nod,inf):
    nx.set_edge_attributes(graph,(1.0-random.random()),'edge_prob')
    nx.set_node_attributes(graph,(1.0-random.random()),'node_prob')
    edge = list(G.edges())
    for i in edge:
        graph[i[0]][i[1]]['mioa_weight'] = -math.log(graph[i[0]][i[1]]['edge_prob'])
    lst=list(G.nodes())
    mioa = []
    for i in lst:
        mioa.append(nx.dijkstra_path(graph,nod,i,weight = 'mioa_weight'))
    return(mioa)


#-----------------------------------------------------------------------------#


#DAG Constructor
def DAG(graph):
    graph.add_node(0)
    l1 = [0]
    l2 = list(itertools.product(l1,G.nodes()))
    l2.pop(-1)
    graph.add_edges_from(l2, weight = 1)
    l3 = MIOA(graph, 0, 0)
    l3 = [i[1:] for i in l3]
    DAG_edges = []
    DAG_nodes = []
    while len(l3[0]) == 1:
        l3.pop(0)
    for i in l3:
        j = 0
        while j < len(i)-1:
            DAG_edges.append([[i[j]],i[j+1]])
    for i in l3:
        for j in i:
            DAG_nodes.append(j)
        
    print("MIOA of DAG:", DAG_edges)
    print("DAG nodes: ", DAG_nodes)


#Algorithm6
S=S1=smax=0
sigma=0
theta=1.0-random.random()
V=G.nodes()
GDAG = nx.read_gml("karate.gml",label='id')
DAG(GDAG)
    