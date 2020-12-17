# -*- coding: utf-8 -*-


import numpy as np
import math
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import scipy as sp

Protein_Atom = ['C','N','O','S']
Ligand_Atom = ['C','N','O','S','P','F','Cl','Br','I']
aa_list = ['ALA','ARG','ASN','ASP','CYS','GLU','GLN','GLY','HIS','HSE','HSD','SEC',
           'ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','PYL']


Year = '2013'
pre = '../data/' + Year

f1 = open(pre +'/name/train_data_' + Year + '.txt')
pre_train_data = f1.readlines()
train_data = eval(pre_train_data[0])
f1.close()

f1 = open(pre + '/name/test_data_' + Year  + '.txt')
pre_test_data = f1.readlines()
test_data = eval(pre_test_data[0])
f1.close()

f1 = open(pre + '/name/all_data_' + Year + '.txt')
pre_all_data = f1.readlines()
all_data = eval(pre_all_data[0])
f1.close()




########################################################################################
# extract coordinate code starts
def get_index(a,b):
    t = len(b)
    if a=='Cl':
        return 6
    if a=='Br':
        return 7
    
    for i in range(t):
        if a[0]==b[i]:
            return i
    return -1


def pocket_coordinate_data_to_file(start,end):
    ####################################################################################
    '''
    this function extract the atom coordinates for each atom-pair of protein-ligand
    complex.
    output is a coordinate file and a description file, the description file records
    the number of atoms for protein and ligand. the coordinate file has four columns, 
    the former three columns are the coordinate, the last column are 1 and 2 for protein
    and ligand atoms respectively. 
    (1) start and end are index of data you will deal with
    (2) before this function, you need to prepare the PDBbind data
    '''
    ####################################################################################
    t1 = len(all_data)
    for i in range(start,end):
        #print('process {0}-th '.format(i))
        
        protein = {}
        for ii in range(4):
            protein[Protein_Atom[ii]] = []
            
        name = all_data[i]
        t1 = pre + '/refined/' + name + '/' + name + '_pocket.pdb'
        f1 = open(t1,'r')
        for line in f1.readlines():
            if (line[0:4]=='ATOM')&(line[17:20] in aa_list ):
                atom = line[13:15]
                atom = atom.strip()
                index = get_index(atom,Protein_Atom)
                if index==-1:
                    continue
                else:
                    protein[Protein_Atom[index]].append(line[30:54])
        f1.close()
        
        
        ligand = {}
        for ii in range(9):
            ligand[Ligand_Atom[ii]] = []
            
        t2 = pre + '/refined/' + name + '/' + name + '_ligand.mol2'
        f2 = open(t2,'r')
        contents = f2.readlines()
        t3 = len(contents)
        start = 0
        end = 0
        for jj in range(t3):
            if contents[jj][0:13]=='@<TRIPOS>ATOM':
                start = jj + 1
                continue
            if contents[jj][0:13]=='@<TRIPOS>BOND':
                end = jj - 1
                break
        for kk in range(start,end+1):
            if contents[kk][8:17]=='thiophene':
                print('thiophene',kk)
            atom = contents[kk][8:10]
            atom = atom.strip()
            index = get_index(atom,Ligand_Atom)
            if index==-1:
                continue
            else:
                    
                ligand[Ligand_Atom[index]].append(contents[kk][17:46])
        f2.close()
        
        
        for i in range(4):
            for j in range(9):
                l_atom = ligand[ Ligand_Atom[j] ]
                p_atom = protein[ Protein_Atom[i] ]
                number_p = len(p_atom)
                number_l = len(l_atom)
                number_all = number_p + number_l
        
                all_atom = np.zeros((number_all,4))
                for jj in range(number_p):
                    all_atom[jj][0] = float(p_atom[jj][0:8])
                    all_atom[jj][1] = float(p_atom[jj][8:16])
                    all_atom[jj][2] = float(p_atom[jj][16:24])
                    all_atom[jj][3] = 1
                for jjj in range(number_p,number_all):
                    all_atom[jjj][0] = float(l_atom[jjj-number_p][0:9])
                    all_atom[jjj][1] = float(l_atom[jjj-number_p][9:19])
                    all_atom[jjj][2] = float(l_atom[jjj-number_p][19:29])
                    all_atom[jjj][3] = 2
        
                filename2 = pre + '/pocket_coordinate/' + name + '_' + Protein_Atom[i] + '_' + Ligand_Atom[j] + '_coordinate.csv'
                np.savetxt(filename2,all_atom,delimiter=',')
                filename3 = pre + '/pocket_coordinate/' + name +  '_' + Protein_Atom[i] + '_' + Ligand_Atom[j] + '_protein_ligand_number.csv'
                temp = np.array(([number_p,number_l]))
                np.savetxt(filename3,temp,delimiter=',')
        

#############################################################################################   
# extract coordinate code ends
                


#######################################################################################################
# create_the_associated_simplicial_complex_of_a_hypergraph algorithm starts 

def distance_of_two_point(p1,p2):
    s = pow(p1[0]-p2[0],2) + pow(p1[1]-p2[1],2) + pow(p1[2]-p2[2],2)
    res = pow(s,0.5)
    return res
    

def get_edge_index1(left,right,edges):
    t = len(edges)
    for i in range(t):
        if (left==edges[i][0])&(right==edges[i][1]):
            return i
    return -1


def create_simplices_with_filtration(atom,cutoff,name,P_atom,L_atom,kill_time):
    
    ###########################################################################################
    ''' 
    this function creates the filtered associated simplicial complex for the hypergraph.
    the dimension only up to 2. you can add higher dimensional information by adding some code. 
    (1) atom is the atom coordinates. the format is same with output of function 
        pocket_coordinate_to_file()
    (2) cutoff determines the binding core region we extract, that is, we extract the ligand
        atoms and the protein atoms within cutoff distance of the ligand. Here, cutoff also 
        determines the largest length of the edges we use to build the hypergraph, here also 
        the associated simplicial complex.(of course you can use many others methods to build 
        the complex, like you can add another parameter max_edge to control the largest length
        of an edge, this is just a way)
    (3) name is the data name.(for example, for PDBbind-2007, it has 1300 data, each data has 
        a name)
    (4) P_atom and L_atom are the atom-combination, like C-C, C-N, etc.
    (5) kill_time is an additional parameter, larger value will lead to longer persistence for
        all the barcode. here we use 0.
    (6) output is a sequence of ordered simplices, i.e. a filtered simplicial complex.
        the format for each simplex is as follows:
        [ index, filtration_value, dimension, vertices of the simplex ]
    '''
    ###########################################################################################
    
    vertices = []
    edge = []
    triangle = []
    edge_same_type = [] # edge_same_type stores the edges come from the same molecular. 
                        # i.e., the edges the hypergraph does not have.
    filtration_of_edge_same_type = []
        
    filename3 = pre + '/pocket_coordinate/' + name + '_' + P_atom + '_' + L_atom +'_protein_ligand_number.csv'
    temp = np.loadtxt(filename3,delimiter=',') # temp gives the numbers of atoms for protein and ligand 
    number_p = int(temp[0])
    number_l = int(temp[1])
   
    t = atom.shape 
    atom_number = t[0] # t is equal to the sum of number_p and number_l
    if (number_p==0)|(number_l==0):# no complex
        return []
    
    for i in range(number_p):
        for j in range(number_p,atom_number):
            dis1 = distance_of_two_point(atom[i],atom[j])
            if dis1<=cutoff:    
                if ([i,j] in edge)==False:
                    edge.append([i,j])
                    if (i in vertices)==False:
                        vertices.append(i)
                    if (j in vertices)==False:
                        vertices.append(j)
                for k in range(atom_number):
                    if (k!=i)&(k!=j):
                        dis = -1
                        if atom[i][3]==atom[k][3]:
                            dis = distance_of_two_point(atom[j],atom[k])
                        else:
                            dis = distance_of_two_point(atom[i],atom[k])
                        
                        if dis<=cutoff:
                            One = 0
                            Two = 0
                            Three = 0
                            if k<i:
                                One = k
                                Two = i
                                Three = j
                            elif (k>i) & (k<j):
                                One = i
                                Two = k
                                Three = j
                            else:
                                One = i
                                Two = j
                                Three = k
                            if ([One,Two,Three] in triangle)==False:
                                triangle.append([One,Two,Three])
                                
                                if ([One,Two] in edge)==False:
                                    edge.append([One,Two])
                                    if atom[One][3]==atom[Two][3]:
                                        edge_same_type.append([One,Two])
                                        d1 = distance_of_two_point(atom[One],atom[Three])
                                        d2 = distance_of_two_point(atom[Two],atom[Three])
                                        d = max(d1,d2)
                                        filtration_of_edge_same_type.append(d)
                                else:
                                    edge_index = get_edge_index1(One,Two,edge_same_type)
                                    if edge_index!=-1:
                                        temp = filtration_of_edge_same_type[edge_index]
                                        d1 = distance_of_two_point(atom[One],atom[Three])
                                        d2 = distance_of_two_point(atom[Two],atom[Three])
                                        d = max(d1,d2)
                                        filtration_of_edge_same_type[edge_index] = max(temp,d)
                            
                                if ([One,Three] in edge)==False:
                                    edge.append([One,Three])
                                    if atom[One][3]==atom[Three][3]:
                                        edge_same_type.append([One,Three])
                                        d1 = distance_of_two_point(atom[One],atom[Two])
                                        d2 = distance_of_two_point(atom[Two],atom[Three])
                                        d = max(d1,d2)
                                        filtration_of_edge_same_type.append(d)
                                else:
                                    edge_index = get_edge_index1(One,Three,edge_same_type)
                                    if edge_index!=-1:
                                        temp = filtration_of_edge_same_type[edge_index]
                                        d1 = distance_of_two_point(atom[One],atom[Two])
                                        d2 = distance_of_two_point(atom[Two],atom[Three])
                                        d = max(d1,d2)
                                        filtration_of_edge_same_type[edge_index] = max(temp,d)
                                    
                                if ([Two,Three] in edge)==False:
                                    edge.append([Two,Three])
                                    if atom[Two][3]==atom[Three][3]:
                                        edge_same_type.append([Two,Three])
                                        d1 = distance_of_two_point(atom[One],atom[Two])
                                        d2 = distance_of_two_point(atom[One],atom[Three])
                                        d = max(d1,d2)
                                        filtration_of_edge_same_type.append(d)
                                else:
                                    edge_index = get_edge_index1(Two,Three,edge_same_type)
                                    if edge_index!=-1:
                                        temp = filtration_of_edge_same_type[edge_index]
                                        d1 = distance_of_two_point(atom[One],atom[Two])
                                        d2 = distance_of_two_point(atom[One],atom[Three])
                                        d = max(d1,d2)
                                        filtration_of_edge_same_type[edge_index] = max(temp,d)
                                    
                                    
                                if (One in vertices)==False:
                                    vertices.append(One)
                                if (Two in vertices)==False:
                                    vertices.append(Two)
                                if (Three in vertices)==False:
                                    vertices.append(Three)
    
    
    for i in range(number_p,atom_number): # here, we add the ligand atoms we did not add in
        if (i in vertices)==False:
            vertices.append(i)
    
    vertices_number = len(vertices)
    edge_number = len(edge)
    triangle_number = len(triangle)
    simplices_with_filtration = []
    
    same_type_number = len(edge_same_type)
    for i in range(same_type_number):
        filtration_of_edge_same_type[i] = filtration_of_edge_same_type[i] + kill_time
   
    if vertices_number==0:
        return []
    for i in range(vertices_number):
        item = [ i , 0 , 0 , vertices[i] ]
        simplices_with_filtration.append(item)
    for i in range( vertices_number , vertices_number + edge_number ):
        one = edge[ i - vertices_number ][0]
        two = edge[ i - vertices_number ][1]
        p1 = atom[ one ]
        p2 = atom[ two ]
        dis = distance_of_two_point(p1,p2)
        edge_index = get_edge_index1(one,two,edge_same_type)
        if edge_index!=-1:
            dis = filtration_of_edge_same_type[edge_index]
        dis = round(dis,15)
        if dis<=cutoff:
            item = [ i , dis , 1 , one , two ]
            simplices_with_filtration.append(item)
    for i in range( vertices_number + edge_number , vertices_number + edge_number + triangle_number ):
        one = triangle[ i - vertices_number - edge_number ][0]
        two = triangle[ i - vertices_number - edge_number ][1]
        three = triangle[ i - vertices_number - edge_number ][2]
        p1 = atom[ one ]
        p2 = atom[ two ]
        p3 = atom[ three ]
        dis = -1
        if ([one,two] in edge_same_type)==False:
            
            dis1 = distance_of_two_point(p1,p2)
            dis = max(dis,dis1)
        else:
            edge_index = get_edge_index1(one,two,edge_same_type)
            temp = filtration_of_edge_same_type[edge_index]
            dis = max(dis,temp)
        if ([one,three] in edge_same_type)==False:
            
            dis2 = distance_of_two_point(p1,p3)
            dis = max(dis,dis2)
        else:
            edge_index = get_edge_index1(one,three,edge_same_type)
            temp = filtration_of_edge_same_type[edge_index]
            dis = max(dis,temp)
        if ([two ,three] in edge_same_type)==False:
            
            dis3 = distance_of_two_point(p2,p3)
            dis = max(dis,dis3)
        else:
            edge_index = get_edge_index1(two,three,edge_same_type)
            temp = filtration_of_edge_same_type[edge_index]
            dis = max(dis,temp)
        dis = round(dis,15)
        if dis<=cutoff:
            item = [ i , dis , 2 , one , two , three ]
            simplices_with_filtration.append(item)
    
    simplices = sorted(simplices_with_filtration,key=lambda x:(x[1]+x[2]/10000000000000000))
    # by applying the function sorted, the simplicies will be ordered by the filtration values.
    # also the face of a simplex will appear earlier than the simplex itself.
    
    for i in range(len(simplices)):
        simplices[i][0] = i # assign index for the ordered simplices
    return simplices


def simplices_to_file(start,end,cutoff,kill_time):
    ################################################################################################
    '''
    this function write the associated simplicial complex of the hypergraph to file
    (1) start and end are the indexes of data we deal with
    (2) cutoff, and kill_time are same with the function "create_simplices_with_filtration"
    (3) before this function, the function pocket_coordinate_data_to_file(start,end) need to 
    be performed to prepare the coordinate data for this function.
    '''
    ################################################################################################
    
    t = len(all_data)
    
    for i in range(start,end):
        name = all_data[i]
        print('process {0}-th data {1}'.format(i,name))
        for P in range(4):
            for L in range(9):
                filename = pre + '/pocket_coordinate/' + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] +'_coordinate.csv'
                point_cloud = np.loadtxt(filename,delimiter=',')
                simplices_with_filtration = create_simplices_with_filtration(point_cloud,cutoff,name,Protein_Atom[P],Ligand_Atom[L],kill_time)
                filename2 = pre + '/pocket_simplices_' + str(cutoff) + '/' + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + str(cutoff) + '.txt'
                f1 = open(filename2,'w')
                f1.writelines(str(simplices_with_filtration))
                f1.close()

                
                
######################################################################################################            
# create_the_associated_simplicial_complex_of_a_hypergraph algorithm ends
                



#########################################################################################################
# from now on, the code is to get the eigenvalue information
def get_point_index(point,points):
    for i in range(len(points)):
        if point==points[i]:
            return i
    
    
def get_edge_index(p1,p2,edges):
    for i in range(len(edges)):
        if (p1==edges[i][0])&(p2==edges[i][1]):
            return i
    

def eigenvalue_of_each_combination_to_file_complex(simplices,name,P,L,cutoff,filtration0,filtration1,grid):
    #print('process {0}-{1} combination of {2}'.format(P,L,name))
    pre1 = pre + '/eigenvalue_' + str(cutoff) + '_' + str(filtration0) + '_' + str(filtration1) + '/'
    
    if len(simplices)==0:
        # no complex, use -1 in the first position as a signal 
        filename1 = pre1 + name + '_' + P + '_' + L + '_' + 'eigenvalue_0D.txt'
        res = [-1]
        f = open(filename1,'w')
        f.writelines(str(res))
        f.close()
        
        filename2 = pre1 + name + '_' + P + '_' + L + '_' + 'eigenvalue_1D.txt'
        res2 = [-1]
        f2 = open(filename2,'w')
        f2.writelines(str(res2))
        f2.close()
        
        return
        
    #get 0-dimension laplacian
    ##################################################################################
    number0 = int((filtration0-2)/grid)
    
    eigenvalue0 = [1] # have complex, use 1 in the first position as a signal
    for i in range(number0 + 1):
        # get eigenvalue for each filtra0 value with grid from 2 to filtration
        filtra0 = 2 + i * grid
        
        points = []
        edges = []
        for r in range(len(simplices)):
            if simplices[r][1]<=filtra0:
                if simplices[r][2]==0:
                    points.append(simplices[r][3])
                elif simplices[r][2]==1:
                    edges.append([ simplices[r][3] , simplices[r][4] ])
                
            else:
                break
        
        row = len(points)
        column = len(edges)
        
        if column==0:
            # only have points, no edges
            res = []
            for ii in range(row):
                res.append(0)
            eigenvalue0.append(res)
            
        else:
            zero_boundary = np.zeros((row,column))
            for j in range(column):
                one = edges[j][0]
                two = edges[j][1]
                index1 = get_point_index(one, points)
                index2 = get_point_index(two, points)
                zero_boundary[index1][j] = -1
                zero_boundary[index2][j] = 1
            Laplacian = np.dot( zero_boundary, zero_boundary.T )
            values = np.linalg.eigvalsh(Laplacian)
            res = []
            for iii in range(len(values)):
                res.append( values[iii] )
            eigenvalue0.append(res)
    
    
    filename1 = pre1 + name + '_' + P + '_' + L + '_' + 'eigenvalue_0D.txt'
    f = open(filename1,'w')
    f.writelines(str(eigenvalue0))
    f.close()
    ##########################################################################################
    
    
    
    #get 1-dimension laplacian
    #############################################################################################
    number1 = int((filtration1-2)/grid)
    
    eigenvalue1 = [1] # have complex, use 1 in the first position as a signal
    for i in range(number1 + 1):
        # get eigenvalue for each filtra1 value with grid from 2 to filtration
        filtra1 = 2 + i * grid
        
        points = []
        edges = []
        triangles = []
        for r in range(len(simplices)):
            if simplices[r][1]<=filtra1:
                if simplices[r][2]==0:
                    points.append(simplices[r][3])
                elif simplices[r][2]==1:
                    edges.append([ simplices[r][3] , simplices[r][4] ])
                elif simplices[r][2]==2:
                    triangles.append( [ simplices[r][3] , simplices[r][4] , simplices[r][5] ] )
                
            else:
                break
        
        N0 = len(points)
        N1 = len(edges)
        N2 = len(triangles)
        
        if N1==0:
            eigenvalue1.append([-1])## have complex, have points but not form edge
            
        elif (N1>0)&(N2==0):
            boundary1 = np.zeros((N0,N1))
            for j in range(N1):
                one = edges[j][0]
                two = edges[j][1]
                index1 = get_point_index(one, points)
                index2 = get_point_index(two, points)
                boundary1[index1][j] = -1
                boundary1[index2][j] = 1
            Laplacian = np.dot( boundary1.T, boundary1 )
            values = np.linalg.eigvalsh(Laplacian)
            res = []
            for iii in range(len(values)):
                res.append( values[iii] )
            eigenvalue1.append(res)
            
        elif (N1>0)&(N2>0):
            
            boundary1 = np.zeros((N0,N1))
            for j in range(N1):
                one = edges[j][0]
                two = edges[j][1]
                index1 = get_point_index(one, points)
                index2 = get_point_index(two, points)
                boundary1[index1][j] = -1
                boundary1[index2][j] = 1
            L1 = np.dot( boundary1.T, boundary1 )
            
            
            boundary2 = np.zeros((N1,N2))
            for j in range(N2):
                one = triangles[j][0]
                two = triangles[j][1]
                three = triangles[j][2]
                index1 = get_edge_index(one, two, edges)
                index2 = get_edge_index(one, three, edges)
                index3 = get_edge_index(two, three, edges)
                boundary2[index1][j] = 1
                boundary2[index2][j] = -1
                boundary2[index3][j] = 1
                
            L2 = np.dot( boundary2, boundary2.T)
            Laplacian = L1 + L2
           
            #[ values , vectors ] = np.linalg.eig(Laplacian)
            values = np.linalg.eigvalsh(Laplacian)
            res = []
            for iii in range(len(values)):
                res.append( values[iii] )
            eigenvalue1.append(res)
    
    
    filename2 = pre1 + name + '_' + P + '_' + L + '_' + 'eigenvalue_1D.txt'
    f = open(filename2,'w')
    f.writelines(str(eigenvalue1))
    f.close()
    ####################################################################################3          
    

def eigenvalue_to_file_complex(start,end,cutoff,filtration0,filtration1,grid):
    for i in range(start,end):
        name = all_data[i]
        print('process {0}-data, {1}'.format(i,name))
        pre1 = pre + '/pocket_simplices_' + str(cutoff) + '/'  
        for P in range(4):
            for L in range(9):
                filename = pre1 + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + str(cutoff) + '.txt'
                f1 = open(filename)
                pre_simplices = f1.readlines()
                simplices = eval(pre_simplices[0])
                f1.close()
                eigenvalue_of_each_combination_to_file_complex(simplices,name,Protein_Atom[P],Ligand_Atom[L],cutoff,filtration0,filtration1,grid)
                






def eigenvalue0_of_each_combination_to_file_ligand(simplices,name,P,L,cutoff,filtration0,grid):
    pre1 = pre + '/ligand_eigenvalue_' + str(cutoff) + '_' + str(filtration0) + '/'
    filename = pre + '/pocket_coordinate/' + name + '_' + P + '_' + L + '_coordinate.csv'
    point_cloud = np.loadtxt(filename,delimiter=',')
    
    if len(simplices)==0:
        # no complex, use -1 in the first position as a signal 
        filename1 = pre1 + name + '_' + P + '_' + L + '_' + 'eigenvalue_0D.txt'
        res = [-1]
        f = open(filename1,'w')
        f.writelines(str(res))
        f.close()
        return
        
    
    #get 0-dimension laplacian, for simplicity, we only consider 0 dimensional information
    number0 = int((filtration0-2)/grid)
    
    eigenvalue0 = [1] # have complex, use 1 in the first position as a signal
    for i in range(number0 + 1):
        # get eigenvalue for each filtra0 value with grid from 2 to filtration
        filtra0 = 2 + i * grid
        
        points = []
        edges = []
        for r in range(len(simplices)):
            if simplices[r][1]<=filtra0:
                if simplices[r][2]==0:
                    temp = simplices[r][3]
                    if point_cloud[temp][3]==2:
                        points.append(simplices[r][3])
                elif simplices[r][2]==1:
                    temp1 = simplices[r][3]
                    temp2 = simplices[r][4]
                    if (point_cloud[temp1][3]==2)&(point_cloud[temp2][3]==2):
                        edges.append([ simplices[r][3] , simplices[r][4] ])
                
            else:
                break
        
        
        row = len(points)
        column = len(edges)
        
        if column==0:
            # only have points, no edges
            res = []
            for ii in range(row):
                res.append(0)
            eigenvalue0.append(res)
            
        else:
            zero_boundary = np.zeros((row,column))
            for j in range(column):
                one = edges[j][0]
                two = edges[j][1]
                index1 = get_point_index(one, points)
                index2 = get_point_index(two, points)
                zero_boundary[index1][j] = -1
                zero_boundary[index2][j] = 1
            Laplacian = np.dot( zero_boundary, zero_boundary.T )
            values = np.linalg.eigvalsh(Laplacian)
            res = []
            for iii in range(len(values)):
                res.append( values[iii] )
            eigenvalue0.append(res)
    
    
    filename1 = pre1 + name + '_' + P + '_' + L + '_' + 'eigenvalue_0D.txt'
    f = open(filename1,'w')
    f.writelines(str(eigenvalue0))
    f.close()
    
    
def eigenvalue_to_file_ligand(start,end,cutoff,filtration0,grid):
    for i in range(start,end):
        
        name = all_data[i]
        print('process {0}-data, {1}'.format(i,name))
        pre1 = pre + '/pocket_simplices_' + str(cutoff) + '/'  
        for P in range(4):
            for L in range(9):
                filename = pre1 + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + str(cutoff) + '.txt'
                f1 = open(filename)
                pre_simplices = f1.readlines()
                simplices = eval(pre_simplices[0])
                f1.close()
                #print(P,L,len(simplices))
                eigenvalue0_of_each_combination_to_file_ligand(simplices,name,Protein_Atom[P],Ligand_Atom[L],cutoff,filtration0,grid)
                
    
#now, get eigenvalue code ends
##############################################################################################################




###########################################################################################################
# feature generation code starts
def get_max(ls):
    if len(ls)==0:
        return 0
    return max(ls)
    
def get_min(ls):
    if len(ls)==0:
         return 0
    return min(ls)
    
def get_mean(ls):
    if len(ls)==0:
        return 0
    return np.mean(ls)
    
def get_std(ls):
    if len(ls)==0:
        return 0
    return np.std(ls)
    
def get_sum(ls):
    if len(ls)==0:
        return 0
    return sum(ls)


def train_feature_to_file(start,end,cutoff,filtration0,filtration1,grid):
    row = end - start
    N = 7
    # N is the number of types of persisitence
    number0 = int ((filtration0-2)/0.1 )
    number1 = int ((filtration1-2)/0.1)
    column = 36 * number0 * N + 36 * number1 * N
    
    feature_matrix = np.zeros((row,column))
    pre1 = pre + '/eigenvalue_' + str(cutoff) + '_' + str(filtration0) + '_' + str(filtration1) + '/'  
    for i in range(start,end):
        print(i)
        name = train_data[i]
        count = 0
        for P in range(4):
            for L in range(9):
                filename0 = pre1 + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'eigenvalue_0D.txt'
                f0 = open(filename0)
                pre_eigenvalue0 = f0.readlines()
                eigenvalue0 = eval(pre_eigenvalue0[0])
                f0.close()
                
                filename1 = pre1 + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'eigenvalue_1D.txt'
                f1 = open(filename1)
                pre_eigenvalue1 = f1.readlines()
                eigenvalue1 = eval(pre_eigenvalue1[0])
                f1.close()
            
                
                #0-D
                ###############################################################################
                if eigenvalue0[0]==-1:
                    for ii in range(number0):
                        for iii in range(N):
                            feature_matrix[i-start][count] = 0
                            count = count + 1
                   
                else:
                    for ii in range(1,number0+1):
                        value = []
                        c0 = 0
                        for iii in range(len(eigenvalue0[ii])):
                            v = eigenvalue0[ii][iii].real
                            if v<=0.000000001: # we take eigenvalues smaller than 0.000000001 as 0  
                                c0 = c0 + 1
                            else:
                                value.append(v)
                        
                        feature_matrix[i-start][count] = c0
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_max(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_min(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_mean(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = len(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_std(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_sum(value)
                        count = count + 1
                        
                       
                ##########################################################################################
                
                #1-D
                ############################################################################################
                if eigenvalue1[0]==-1:
                    for ii in range(number1):
                        for iii in range(N):
                            feature_matrix[i-start][count] = 0
                            count = count + 1
                           
                else:
                    for ii in range(1,number1+1):
                            if (eigenvalue1[ii][0]==-1): ##have point but have not form edge
                                for iii in range(N):
                                    feature_matrix[i-start][count] = 0
                                    count = count + 1
                                continue
                            
                            
                            value = []
                            c0 = 0
                            for iii in range(len(eigenvalue1[ii])):
                                v = eigenvalue1[ii][iii].real
                                if v<=0.000000001:
                                    c0 = c0 + 1
                                else:
                                    value.append(v)
                            
                            feature_matrix[i-start][count] = c0
                            count = count + 1
                            
                            feature_matrix[i-start][count] = get_max(value)
                            count = count + 1
                            
                            feature_matrix[i-start][count] = get_min(value)
                            count = count + 1
                            
                            feature_matrix[i-start][count] = get_mean(value)
                            count = count + 1
                            
                            feature_matrix[i-start][count] = len(value)
                            count = count + 1
                            
                            feature_matrix[i-start][count] = get_std(value)
                            count = count + 1
                            
                            feature_matrix[i-start][count] = get_sum(value)
                            count = count + 1
                            
                 ###########################################################################################
                        
    filename = pre + '/feature/' + 'laplacian_train_' + str(cutoff) + '_' + str(filtration0) + '_' + str(filtration1) + '_' + str(grid) + '.csv'
    np.savetxt(filename,feature_matrix,delimiter=',')
    
    
def test_feature_to_file(start,end,cutoff,filtration0,filtration1,grid):
    row = end - start
    N = 7
    # N is the number of types of persisitence
    number0 = int ((filtration0-2)/0.1 )
    number1 = int ((filtration1-2)/0.1)
    column = 36 * number0 * N + 36 * number1 * N
    
    feature_matrix = np.zeros((row,column))
    pre1 = pre + '/eigenvalue_' + str(cutoff) + '_' + str(filtration0) + '_' + str(filtration1) + '/'  
    for i in range(start,end):
        print(i)
        name = test_data[i]
        count = 0
        for P in range(4):
            for L in range(9):
                filename0 = pre1 + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'eigenvalue_0D.txt'
                f0 = open(filename0)
                pre_eigenvalue0 = f0.readlines()
                eigenvalue0 = eval(pre_eigenvalue0[0])
                f0.close()
                
                filename1 = pre1 + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'eigenvalue_1D.txt'
                f1 = open(filename1)
                pre_eigenvalue1 = f1.readlines()
                eigenvalue1 = eval(pre_eigenvalue1[0])
                f1.close()
                
                
                #0-D
                ###############################################################################
                if eigenvalue0[0]==-1:
                    for ii in range(number0):
                        for iii in range(N):
                            feature_matrix[i-start][count] = 0
                            count = count + 1
                else:
                   
                    for ii in range(1,number0+1):
                        value = []
                        c0 = 0
                        for iii in range(len(eigenvalue0[ii])):
                            v = eigenvalue0[ii][iii].real
                            if v<=0.000000001:
                                c0 = c0 + 1
                            else:
                                value.append(v)
                    
                        feature_matrix[i-start][count] = c0
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_max(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_min(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_mean(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = len(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_std(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_sum(value)
                        count = count + 1
                        
                ##########################################################################################
                
                #1-D
                ############################################################################################
                if eigenvalue1[0]==-1:
                    for ii in range(number1):
                        for iii in range(N):
                            feature_matrix[i-start][count] = 0
                            count = count + 1
                           
                else:
                    for ii in range(1,number1+1):
                            if (eigenvalue1[ii][0]==-1): #have point but have not form edge
                                for iii in range(N):
                                    feature_matrix[i-start][count] = 0
                                    count = count + 1
                                continue
                            
                            
                            value = []
                            c0 = 0
                            for iii in range(len(eigenvalue1[ii])):
                                v = eigenvalue1[ii][iii].real
                                if v<=0.000000001:
                                    c0 = c0 + 1
                                else:
                                    value.append(v)
                            
                            feature_matrix[i-start][count] = c0
                            count = count + 1
                            
                            feature_matrix[i-start][count] = get_max(value)
                            count = count + 1
                            
                            feature_matrix[i-start][count] = get_min(value)
                            count = count + 1
                            
                            feature_matrix[i-start][count] = get_mean(value)
                            count = count + 1
                            
                            feature_matrix[i-start][count] = len(value)
                            count = count + 1
                            
                            feature_matrix[i-start][count] = get_std(value)
                            count = count + 1
                           
                            feature_matrix[i-start][count] = get_sum(value)
                            count = count + 1
                            
                 ###########################################################################################
                        
    filename = pre + '/feature/' + 'laplacian_test_' + str(cutoff) + '_' + str(filtration0) + '_' + str(filtration1) + '_' + str(grid) + '.csv'
    np.savetxt(filename,feature_matrix,delimiter=',')


def train_feature_to_file_combined(start,end,cutoff,filtration0,filtration1,grid):
    row = end - start
    N = 7
    # N is the number of types of persisitence
    number0 = int ((filtration0-2)/0.1 )
    number1 = int ((filtration1-2)/0.1 )
    column = 36 * number0 * N + 36 * number1 * N + 36 * number0 * N 
    
    feature_matrix = np.zeros((row,column))
    pre0 =  pre + '/eigenvalue_' + str(cutoff) + '_' + str(filtration0) + '_' + str(filtration1) + '/' 
    pre1 =  pre + '/ligand_eigenvalue_' + str(cutoff) + '_' + str(filtration0) + '/'
    
    for i in range(start,end):
        print(i)
        name = train_data[i]
        count = 0
        for P in range(4):
            for L in range(9):
                filename0 = pre0 + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'eigenvalue_0D.txt'
                f0 = open(filename0)
                pre_eigenvalue0 = f0.readlines()
                eigenvalue0 = eval(pre_eigenvalue0[0])
                f0.close()
                
                filename1 = pre0 + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'eigenvalue_1D.txt'
                f1 = open(filename1)
                pre_eigenvalue1 = f1.readlines()
                eigenvalue1 = eval(pre_eigenvalue1[0])
                f1.close()
                
                filename2 = pre1 + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'eigenvalue_0D.txt'
                f2 = open(filename2)
                pre_sub_eigenvalue0 = f2.readlines()
                sub_eigenvalue0 = eval(pre_sub_eigenvalue0[0])
                f2.close()
                
                
               
                
                if eigenvalue0[0]==-1:
                    for ii in range(number0):
                        for iii in range(N):
                            feature_matrix[i-start][count] = 0
                            count = count + 1
                    
                    for ii in range(number0): # ligand
                        for iii in range(N):
                            feature_matrix[i-start][count] = 0
                            count = count + 1
                   
                else:
                    
                    for ii in range(1,number0+1):
                        value = []
                        c0 = 0
                        for iii in range(len(eigenvalue0[ii])):
                            v = eigenvalue0[ii][iii]
                            if v<=0.000000001:
                                c0 = c0 + 1
                            else:
                                value.append(v)
                        
                        feature_matrix[i-start][count] = c0
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_max(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_min(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_mean(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = len(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_std(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_sum(value)
                        count = count + 1
                        
                      
                    for ii in range(1,number0+1): # ligand
                        value = []
                        c0 = 0
                        for iii in range(len(sub_eigenvalue0[ii])):
                            v = sub_eigenvalue0[ii][iii]
                            if v<=0.000000001:
                                c0 = c0 + 1
                            else:
                                value.append(v)
                        
                        feature_matrix[i-start][count] = c0
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_max(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_min(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_mean(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = len(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_std(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_sum(value)
                        count = count + 1
                        
                      
                ##########################################################################################
                        
                if eigenvalue1[0]==-1:
                    for ii in range(number1):
                        for jj in range(N):
                            feature_matrix[i-start][count] = 0
                            count = count + 1
                else:
                    for ii in range(1,number1+1):
                            if (eigenvalue1[ii][0]==-1): ##have point but have not form edge
                                for iii in range(N):
                                    feature_matrix[i-start][count] = 0
                                    count = count + 1
                                continue
                            
                            
                            value = []
                            c0 = 0
                            for iii in range(len(eigenvalue1[ii])):
                                v = eigenvalue1[ii][iii]
                                if v<=0.000000001:
                                    c0 = c0 + 1
                                else:
                                    value.append(v)
                            #print(value)
                            feature_matrix[i-start][count] = c0
                            count = count + 1
                            
                            feature_matrix[i-start][count] = get_max(value)
                            count = count + 1
                            
                            feature_matrix[i-start][count] = get_min(value)
                            count = count + 1
                            
                            feature_matrix[i-start][count] = get_mean(value)
                            count = count + 1
                            
                            feature_matrix[i-start][count] = len(value)
                            count = count + 1
                            
                            feature_matrix[i-start][count] = get_std(value)
                            count = count + 1
                            
                            feature_matrix[i-start][count] = get_sum(value) 
                            count = count + 1
                            
                           
                 ###########################################################################################
                    
    filename = pre + '/feature/' + 'combined_laplacian_train_' + str(cutoff) + '_' + str(filtration0) + '_' + str(filtration1) + '_' + str(grid) + '.csv'
    np.savetxt(filename,feature_matrix,delimiter=',')


def test_feature_to_file_combined(start,end,cutoff,filtration0,filtration1,grid):
    row = end - start
    N = 7
    # N is the number of types of persisitence
    number0 = int ((filtration0-2)/0.1 )
    number1 = int ((filtration1-2)/0.1 )
    column = 36 * number0 * N + 36 * number1 * N + 36 * number0 * N 
    
    feature_matrix = np.zeros((row,column))
    pre0 =  pre + '/eigenvalue_' + str(cutoff) + '_' + str(filtration0) + '_' + str(filtration1) + '/' 
    pre1 =  pre + '/ligand_eigenvalue_' + str(cutoff) + '_' + str(filtration0) + '/'
    
    for i in range(start,end):
        print(i)
        name = test_data[i]
        count = 0
        for P in range(4):
            for L in range(9):
                filename0 = pre0 + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'eigenvalue_0D.txt'
                f0 = open(filename0)
                pre_eigenvalue0 = f0.readlines()
                eigenvalue0 = eval(pre_eigenvalue0[0])
                f0.close()
                
                filename1 = pre0 + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'eigenvalue_1D.txt'
                f1 = open(filename1)
                pre_eigenvalue1 = f1.readlines()
                eigenvalue1 = eval(pre_eigenvalue1[0])
                f1.close()
                
                filename2 = pre1 + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'eigenvalue_0D.txt'
                f2 = open(filename2)
                pre_sub_eigenvalue0 = f2.readlines()
                sub_eigenvalue0 = eval(pre_sub_eigenvalue0[0])
                f2.close()
                
                if eigenvalue0[0]==-1:
                    for ii in range(number0):
                        for iii in range(N):
                            feature_matrix[i-start][count] = 0
                            count = count + 1
                    
                    for ii in range(number0): # ligand
                        for iii in range(N):
                            feature_matrix[i-start][count] = 0
                            count = count + 1
                   
                else:
                    
                    for ii in range(1,number0+1):
                        value = []
                        c0 = 0
                        for iii in range(len(eigenvalue0[ii])):
                            v = eigenvalue0[ii][iii]
                            if v<=0.000000001:
                                c0 = c0 + 1
                            else:
                                value.append(v)
                        
                        feature_matrix[i-start][count] = c0
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_max(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_min(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_mean(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = len(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_std(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_sum(value)
                        count = count + 1
                        
                      
                    for ii in range(1,number0+1): # ligand
                        value = []
                        c0 = 0
                        for iii in range(len(sub_eigenvalue0[ii])):
                            v = sub_eigenvalue0[ii][iii]
                            if v<=0.000000001:
                                c0 = c0 + 1
                            else:
                                value.append(v)
                        
                        feature_matrix[i-start][count] = c0
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_max(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_min(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_mean(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = len(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_std(value)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_sum(value)
                        count = count + 1
                        
                      
                ##########################################################################################
                        
                if eigenvalue1[0]==-1:
                    for ii in range(number1):
                        for jj in range(N):
                            feature_matrix[i-start][count] = 0
                            count = count + 1
                else:
                    for ii in range(1,number1+1):
                            if (eigenvalue1[ii][0]==-1): ##have point but have not form edge
                                for iii in range(N):
                                    feature_matrix[i-start][count] = 0
                                    count = count + 1
                                continue
                            
                            value = []
                            c0 = 0
                            for iii in range(len(eigenvalue1[ii])):
                                v = eigenvalue1[ii][iii]
                                if v<=0.000000001:
                                    c0 = c0 + 1
                                else:
                                    value.append(v)
                            #print(value)
                            feature_matrix[i-start][count] = c0
                            count = count + 1
                            
                            feature_matrix[i-start][count] = get_max(value)
                            count = count + 1
                            
                            feature_matrix[i-start][count] = get_min(value)
                            count = count + 1
                            
                            feature_matrix[i-start][count] = get_mean(value)
                            count = count + 1
                            
                            feature_matrix[i-start][count] = len(value)
                            count = count + 1
                            
                            feature_matrix[i-start][count] = get_std(value)
                            count = count + 1
                            
                            feature_matrix[i-start][count] = get_sum(value) 
                            count = count + 1
                            
                           
                 ###########################################################################################
                    
    filename = pre + '/feature/' + 'combined_laplacian_test_' + str(cutoff) + '_' + str(filtration0) + '_' + str(filtration1) + '_' + str(grid) + '.csv'
    np.savetxt(filename,feature_matrix,delimiter=',')


def get_name_index(name,contents):
    t = len(contents)
    for i in range(t):
        if contents[i][0:4]==name:
            return i


def get_target_matrix_of_train():
    t = len(train_data)
    target_matrix = []
    t1 = pre + '/' + Year + '_INDEX_refined.data'
    f1 = open(t1,'r')
    contents = f1.readlines()
    f1.close()
    for i in range(t):  # tttttttttttttttttttttttttttttttttt
        name = train_data[i]
        index = get_name_index(name,contents)
        target_matrix.append(float(contents[index][18:23]))
    res = np.array(target_matrix)
    np.savetxt(pre + '/feature/' + 'target_matrix_of_train.csv',res,delimiter=',')


def get_target_matrix_of_test():
    t = len(test_data)
    target_matrix = []
    t1 = pre + '/' + Year + '_INDEX_refined.data'
    f1 = open(t1,'r')
    contents = f1.readlines()
    f1.close()
    for i in range(t):  # tttttttttttttttttttttttttttttttttt
        name = test_data[i]
        index = get_name_index(name,contents)
        target_matrix.append(float(contents[index][18:23]))
    res = np.array(target_matrix)
    np.savetxt(pre + '/feature/' + 'target_matrix_of_test.csv',res,delimiter=',')
    




# feature generation code ends.
###########################################################################################################




############################################################################################################
# machine_learning algorithm starts.
    
def gradient_boosting(X_train,Y_train,X_test,Y_test):
    params={'n_estimators': 40000, 'max_depth': 8, 'min_samples_split': 2,
                'learning_rate': 0.001, 'loss': 'ls','max_features':'sqrt','subsample':0.7}
    regr = GradientBoostingRegressor(**params)
    regr.fit(X_train,Y_train)
    pearson_coorelation = sp.stats.pearsonr(Y_test,regr.predict(X_test))
    mse1 = mean_squared_error(Y_test, regr.predict(X_test))
    mse2 = pow(mse1,0.5)
    #mse3 = mse2/0.7335
    mse3 = mse2
    return [pearson_coorelation[0],mse3]

def get_pearson_correlation(typ,pref):
    feature_matrix_of_train = np.loadtxt( pre + '/feature/' + pref +'laplacian_train_10_7_7_0.1.csv',delimiter=',' )
    target_matrix_of_train = np.loadtxt( pre + '/feature/' + 'target_matrix_of_train.csv',delimiter=',' )
    feature_matrix_of_test = np.loadtxt( pre + '/feature/' + pref + 'laplacian_test_10_7_7_0.1.csv',delimiter=',' )
    target_matrix_of_test = np.loadtxt( pre + '/feature/' +  'target_matrix_of_test.csv',delimiter=',' )
    number = 10
    P = np.zeros((number,1))
    M = np.zeros((number,1))
    #print(feature_matrix_of_test.shape)
    for i in range(number):
        [P[i][0],M[i][0]] = gradient_boosting(feature_matrix_of_train,target_matrix_of_train,feature_matrix_of_test,target_matrix_of_test)
        print(P[i])
    median_p = np.median(P)
    median_m = np.median(M)
    print('for data ' + Year + ', 10 results for ' + typ + '-model are:')
    print(P)
    print('median pearson correlation values are')
    print(median_p)
    print('median mean squared error values are')
    print(median_m)
    
    
############################################################################################################
# machine_learning algorithm ends.


def run_for_PDBbind_2013():
    ##############################################################
    '''
    by running this function, you can get the results for data2013
    '''
    ##############################################################
    
    
    # extract coordinate
    pocket_coordinate_data_to_file(0,2959) 
    
    # create hypergraph
    simplices_to_file(0,2959,10,0)      
    
    # compute_spectral_information
    eigenvalue_to_file_complex(0,2959,10,7,7,0.1)
    eigenvalue_to_file_ligand(0,2959,10,7,0.1)
    
    # feature generation
    train_feature_to_file(0,2764,10,7,7,0.1)
    test_feature_to_file(0,195,10,7,7,0.1)
    train_feature_to_file_combined(0,2764,10,7,7,0.1)
    test_feature_to_file_combined(0,195,10,7,7,0.1)
    
    get_target_matrix_of_train()
    get_target_matrix_of_test()
    
    # machine learning
    get_pearson_correlation('complex','')
    get_pearson_correlation('combined','combined_')
    
    
run_for_PDBbind_2013()


