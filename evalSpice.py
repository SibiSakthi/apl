import numpy as np
import io
def input_manupulation(filename):
    sfp = io.StringIO(filename)
    dict={}
    word=sfp.readlines()
    x1=0
    m1=0
    for x in word:
        word1=x.strip()
        if x1 == 0:
            if word1 != ".circuit":
                continue
            else:
                x1 = 1
                continue
        if (word1==".end"):
            x1=2
            break
        word1=word1.split()
        try:
            word1.remove("dc")
        except:
               pass    
        try:   
            for x5 in range (4):
                if word1[x5]=="GND":
                    word1[x5]=0
                elif  word1[x5][0]=='n':
                    word1[x5]=int(word1[x5][1:] )
            m1=max(int(word1[1]),int(word1[2]),m1)         
            if ( word1[0][0] in dict ):
                dict[word1[0][0] ].append([word1[1],word1[2],word1[3]])
            else :
                dict[word1[0][0]] = [[word1[1],word1[2],word1[3]]]
        except :
            pass
    if (x1!=2) :
        raise  ValueError('Malformed circuit file')
    else :
        for key in dict :
            if (key=='R'or key=='I' or key=='V'):
                pass
            else :
                raise ValueError('Only V, I, R elements are permitted') 
    return dict,m1
def matrix_maker(dict,m1):
    try:
        num_nodes=m1+len (dict['V'])+1
    except :
        num_nodes=m1+1
    A = np.zeros((num_nodes, num_nodes))
    B = np.zeros((num_nodes, 1))
    m1=m1+1
    for x1 in dict:
        if x1 =="R":
            for x2 in dict[x1]:
                node1 = int(x2[0])   # Adjust for 0-based indexing
                node2 = int(x2[1]) 
                value = 1/float(x2[2])
                A[node1, node1] +=  value
                A[node2, node2] +=  value
                A[node1, node2] -=  value
                A[node2, node1] -=  value
        elif x1 =="I":
            for x2 in dict[x1]:
                node1 = int(x2[0])  # Adjust for 0-based indexing
                node2 = int(x2[1]) 
                value = float(x2[2])
                B[node1, 0] -= value
                B[node2, 0] += value
        elif x1 =="V":
            for x2 in dict[x1]:
                node1 = int(x2[0])   # Adjust for 0-based indexing
                node2 = int(x2[1]) 
                A[node1,m1]=1
                A[node2,m1]=-1
                A[m1,node1]=1
                A[m1,node2]=-1
                B[m1]=float(x2[2])
                m1=m1+1
    A = A[1: num_nodes, 1: num_nodes]
    B = B[1: num_nodes]
    node_voltages = np.linalg.solve(A, B)
    return node_voltages,m1-1

            
def evalSpice(filename):
    volt_dict={}
    current_dict={}
    try :
        text_file = open(filename , "r")
        data = text_file.read()
        text_file.close()
    except :
        raise FileNotFoundError('Please give the name of a valid SPICE file as input')    
    x1,x2=input_manupulation(data)
    m1,num_nodes=matrix_maker(x1,x2)
    if np.isnan(m1).any() :
        raise  AssertionError('Circuit error: no solution')
    else:
        volt_dict['GND']=0.0
        for i in range (1,num_nodes):
            volt_dict['n'+ str(i)]=float(m1[i-1])
        try:
            current_dict['Vs']=float(m1[num_nodes-1])
        except :
            pass
    return (volt_dict, current_dict)
#testdata = "./testdata/"
#print(evalSpice(testdata + "test_1.ckt"))