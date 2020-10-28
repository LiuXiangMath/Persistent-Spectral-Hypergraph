# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:17:55 2020

@author: liuxiang
"""

import numpy as np
import math
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import scipy as sp

Protein_Atom = ['C','N','O','S']
Ligand_Atom = ['C','N','O','S','P','F','Cl','Br','I']


Year = '2007'

f1 = open(Year +'/name/train_data_' + Year + '.txt')
pre_train_data = f1.readlines()
train_data = eval(pre_train_data[0])
f1.close()

f1 = open(Year + '/name/test_data_' + Year  + '.txt')
pre_test_data = f1.readlines()
test_data = eval(pre_test_data[0])
f1.close()

f1 = open(Year + '/name/all_data_' + Year + '.txt')
pre_all_data = f1.readlines()
all_data = eval(pre_all_data[0])
f1.close()


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
    pre = Year + '/eigenvalue_' + str(cutoff) + '_' + str(filtration0) + '_' + str(filtration1) + '/'
    
    
    if len(simplices)==0:
        # no complex, use -1 in the first position as a signal 
        filename1 = pre + name + '_' + P + '_' + L + '_' + 'eigenvalue_0D.txt'
        res = [-1]
        f = open(filename1,'w')
        f.writelines(str(res))
        f.close()
        
        filename2 = pre + name + '_' + P + '_' + L + '_' + 'eigenvalue_1D.txt'
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
        #filtra0 = 1
        
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
            
            #[ values , vectors ] = np.linalg.eig(Laplacian)
            values = np.linalg.eigvalsh(Laplacian)
            res = []
            for iii in range(len(values)):
                res.append( values[iii] )
            eigenvalue0.append(res)
    
    
    filename1 = pre + name + '_' + P + '_' + L + '_' + 'eigenvalue_0D.txt'
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
        #filtra1 = 4
        
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
            
            #[ values , vectors ] = np.linalg.eig(Laplacian)
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
    
    
    filename2 = pre + name + '_' + P + '_' + L + '_' + 'eigenvalue_1D.txt'
    f = open(filename2,'w')
    f.writelines(str(eigenvalue1))
    f.close()
    ####################################################################################3          
    


def eigenvalue_to_file_complex(start,end,cutoff,filtration0,filtration1,grid):
    for i in range(start,end):
        name = all_data[i]
        print('process {0}-data, {1}'.format(i,name))
        pre = Year + '/pocket_simplices_' + str(cutoff) + '/'  
        for P in range(4):
            for L in range(9):
                filename = pre + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + str(cutoff) + '.txt'
                f1 = open(filename)
                pre_simplices = f1.readlines()
                simplices = eval(pre_simplices[0])
                f1.close()
                eigenvalue_of_each_combination_to_file_complex(simplices,name,Protein_Atom[P],Ligand_Atom[L],cutoff,filtration0,filtration1,grid)
                
                
                
#eigenvalue_to_file_complex(0,1300,10,7,7,0.1)






def eigenvalue0_of_each_combination_to_file_ligand(simplices,name,P,L,cutoff,filtration0,filtration1,grid):
    pre = Year + '/ligand_eigenvalue_' + str(cutoff) + '_' + str(filtration0) + '_' + str(filtration1) + '/'
    
    
    filename = Year + '/' + Year + '_pocket_coordinate/' + name + '_' + P + '_' + L + '_coordinate.csv'
    point_cloud = np.loadtxt(filename,delimiter=',')
    
    if len(simplices)==0:
        # no complex, use -1 in the first position as a signal 
        filename1 = pre + name + '_' + P + '_' + L + '_' + 'eigenvalue_0D.txt'
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
            
            #[ values , vectors ] = np.linalg.eig(Laplacian)
            values = np.linalg.eigvalsh(Laplacian)
            res = []
            for iii in range(len(values)):
                res.append( values[iii] )
            eigenvalue0.append(res)
    
    
    filename1 = pre + name + '_' + P + '_' + L + '_' + 'eigenvalue_0D.txt'
    f = open(filename1,'w')
    f.writelines(str(eigenvalue0))
    f.close()
    
def eigenvalue_to_file_ligand(start,end,cutoff,filtration0,filtration1,grid):
    for i in range(start,end):
        
        name = all_data[i]
        print('process {0}-data, {1}'.format(i,name))
        pre = Year + '/pocket_simplices_' + str(cutoff) + '/'  
        for P in range(4):
            for L in range(9):
                filename = pre + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + str(cutoff) + '.txt'
                f1 = open(filename)
                pre_simplices = f1.readlines()
                simplices = eval(pre_simplices[0])
                f1.close()
                #print(P,L,len(simplices))
                eigenvalue0_of_each_combination_to_file_ligand(simplices,name,Protein_Atom[P],Ligand_Atom[L],cutoff,filtration0,filtration1,grid)
                
    
#eigenvalue_to_file_subhypergraph(0,1300,10,7,7,0.1)

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
    pre = Year + '/eigenvalue_' + str(cutoff) + '_' + str(filtration0) + '_' + str(filtration1) + '/'  
    for i in range(start,end):
        print(i)
        name = train_data[i]
        count = 0
        for P in range(4):
            for L in range(9):
                filename0 = pre + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'eigenvalue_0D.txt'
                f0 = open(filename0)
                pre_eigenvalue0 = f0.readlines()
                eigenvalue0 = eval(pre_eigenvalue0[0])
                f0.close()
                
                filename1 = pre + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'eigenvalue_1D.txt'
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
                        
    filename = Year + '/feature/' + str(start) + '_' + str(end) + '_' + 'laplacian_train_' + str(cutoff) + '_' + str(filtration0) + '_' + str(filtration1) + '_' + str(grid) + '.csv'
    np.savetxt(filename,feature_matrix,delimiter=',')
    
    
    
    
def test_feature_to_file(start,end,cutoff,filtration0,filtration1,grid):
    row = end - start
    N = 7
    # N is the number of types of persisitence
    number0 = int ((filtration0-2)/0.1 )
    number1 = int ((filtration1-2)/0.1)
    column = 36 * number0 * N + 36 * number1 * N
    
    feature_matrix = np.zeros((row,column))
    pre = Year + '/eigenvalue_' + str(cutoff) + '_' + str(filtration0) + '_' + str(filtration1) + '/'  
    for i in range(start,end):
        print(i)
        name = test_data[i]
        count = 0
        for P in range(4):
            for L in range(9):
                filename0 = pre + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'eigenvalue_0D.txt'
                f0 = open(filename0)
                pre_eigenvalue0 = f0.readlines()
                eigenvalue0 = eval(pre_eigenvalue0[0])
                f0.close()
                
                filename1 = pre + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'eigenvalue_1D.txt'
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
                        
    filename = Year + '/feature/' + str(start) + '_' + str(end) + '_' + 'laplacian_test_' + str(cutoff) + '_' + str(filtration0) + '_' + str(filtration1) + '_' + str(grid) + '.csv'
    np.savetxt(filename,feature_matrix,delimiter=',')


                
                
#train_feature_to_file(0,1105,10,7,7,0.1)
#test_feature_to_file(0,195,10,7,7,0.1)


def train_feature_to_file_combined(start,end,cutoff,filtration0,filtration1,grid):
    row = end - start
    N = 7
    # N is the number of types of persisitence
    number0 = int ((filtration0-2)/0.1 )
    number1 = int ((filtration1-2)/0.1 )
    column = 36 * number0 * N + 36 * number1 * N + 36 * number0 * N 
    
    feature_matrix = np.zeros((row,column))
    pre0 =  Year + '/eigenvalue_' + str(cutoff) + '_' + str(filtration0) + '_' + str(filtration1) + '/' 
    pre1 =  Year + '/ligand_eigenvalue_' + str(cutoff) + '_' + str(filtration0) + '_' + str(filtration1) + '/'
    
    
    
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
                    
                        
                        
    filename = Year + '/feature/' + str(start) + '_' + str(end) + '_' + 'combined_laplacian_train_' + str(cutoff) + '_' + str(filtration0) + '_' + str(filtration1) + '_' + str(grid) + '.csv'
    np.savetxt(filename,feature_matrix,delimiter=',')



def test_feature_to_file_combined(start,end,cutoff,filtration0,filtration1,grid):
    row = end - start
    N = 7
    # N is the number of types of persisitence
    number0 = int ((filtration0-2)/0.1 )
    number1 = int ((filtration1-2)/0.1 )
    column = 36 * number0 * N + 36 * number1 * N + 36 * number0 * N 
    
    feature_matrix = np.zeros((row,column))
    pre0 =  Year + '/eigenvalue_' + str(cutoff) + '_' + str(filtration0) + '_' + str(filtration1) + '/' 
    pre1 =  Year + '/ligand_eigenvalue_' + str(cutoff) + '_' + str(filtration0) + '_' + str(filtration1) + '/'
    
    
    
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
                    
                        
                        
    filename = Year + '/feature/' + str(start) + '_' + str(end) + '_' + 'combined_laplacian_test_' + str(cutoff) + '_' + str(filtration0) + '_' + str(filtration1) + '_' + str(grid) + '.csv'
    np.savetxt(filename,feature_matrix,delimiter=',')



#combined_train_feature_to_file(0,1105,10,7,7,0.1)
#combined_test_feature_to_file(0,195,10,7,7,0.1)


# feature generation code ends.
###########################################################################################################




#######################################################################################################
# machine learning starts

def gradient_boosting(X_train,Y_train,X_test,Y_test,max_depth,subsample):
    params={'n_estimators': 40000, 'max_depth': max_depth, 'min_samples_split': 2,
                'learning_rate': 0.001, 'loss': 'ls','max_features':'sqrt','subsample':subsample}
    regr = GradientBoostingRegressor(**params)
    regr.fit(X_train,Y_train)
    pearson_coorelation = sp.stats.pearsonr(Y_test,regr.predict(X_test))
    mse1 = mean_squared_error(Y_test, regr.predict(X_test))
    mse2 = pow(mse1,0.5)
    mse3 = mse2
    return [pearson_coorelation[0],mse3]


def get_pearson_correlation(max_depth,subsample):
    filename = Year + '/feature/'
    feature_matrix_of_train = np.loadtxt( filename + 'laplacian_train_10_7_7_0.1.csv',delimiter=',' )
    target_matrix_of_train = np.loadtxt( filename + 'target_matrix_of_train.csv',delimiter=',' )
    feature_matrix_of_test = np.loadtxt( filename + 'laplacian_test_10_7_7_0.1.csv',delimiter=',' )
    target_matrix_of_test = np.loadtxt( filename +  'target_matrix_of_test.csv',delimiter=',' )
    number = 10
    P = np.zeros((number,1))
    M = np.zeros((number,1))
    print(feature_matrix_of_test.shape)
    for i in range(number):
        
        [P[i][0],M[i][0]] = gradient_boosting(feature_matrix_of_train,target_matrix_of_train,feature_matrix_of_test,target_matrix_of_test,max_depth,subsample)
        print([P[i][0],M[i][0]])
    median_p = np.median(P)
    median_m = np.median(M)
    print('median pearson correlation values are')
    #print(P)
    print(median_p)
    print('median mean squared error values are')
    #print(M)
    print(median_m)


#get_pearson_correlation(8,0.7)
    

# machine learning ends
##################################################################################################





