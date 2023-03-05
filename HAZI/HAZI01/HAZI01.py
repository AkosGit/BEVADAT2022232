# %%
#Create a function that returns with a subsest of a list.
#The subset's starting and ending indexes should be set as input parameters (the list aswell).
#return type: list
#function name must be: subset
#input parameters: input_list,start_index,end_index

# %%
def subset(input_list,start_index,end_index):
    out=[]
    for i in range(len(input_list)):
        if(i >= (start_index) and i<= (end_index-1)):
            out.append(input_list[i])
    return out
#subset([1,2,3,4,5], 1,3)

# %%
#Create a function that returns every nth element of a list.
#return type: list
#function name must be: every_nth
#input parameters: input_list,step_size

# %%
def every_nth(input_list,step_size):
    out=[]
    for i in range(len(input_list)):
        if (i)%step_size==0 or i is 0:
            out.append(input_list[i])
    return out
#every_nth([1,2,3,4,5,6,7,8,9],3)

# %%
#Create a function that can decide whether a list contains unique values or not
#return type: bool
#function name must be: unique
#input parameters: input_list

# %%
def unique(input_list):
    out=[]
    for e in input_list:
        if e not in out:
            out.append(e)
        else:
            return False
    return True
#unique([1,0,1])

# %%
#Create a function that can flatten a nested list ([[..],[..],..])
#return type: list
#fucntion name must be: flatten
#input parameters: input_list

# %%
def flatten(input_list):
    out=[]
    for l in input_list:
        for e in l:
            out.append(e)
    return out
#flatten([[1,2],[3,4],[5,6]])

# %%
#Create a function that concatenates n lists
#return type: list
#function name must be: merge_lists
#input parameters: *args


# %%
def merge_lists(*args):
    out=[]
    for l in args:
        out.extend(l)
    return out
#merge_lists([1,2],[3,4],[5,6])

# %%
#Create a function that can reverse a list of tuples
#example [(1,2),...] => [(2,1),...]
#return type: list
#fucntion name must be: reverse_tuples
#input parameters: input_list

# %%
def reverse_tuples(input_list):
    for i in range(len(input_list)):
        l=[]
        for j in reversed(range(len(input_list[i]))):
            l.append(input_list[i][j])
        input_list[i]=tuple(l)
    return input_list
#reverse_tuples([(1,2), (3,4)])

# %%
#Create a function that removes duplicates from a list
#return type: list
#fucntion name must be: remove_tuplicates
#input parameters: input_list

# %%
def remove_duplicates(input_list):
    out=[]
    for e in input_list:
        if e not in out:
            out.append(e)
    return out
#remove_tuplicates([1,2,2,3,4,4]) 

# %%
#Create a function that transposes a nested list (matrix)
#return type: list
#function name must be: transpose
#input parameters: input_list

# %%
def transpose(input_list):
    out=[]
    for i in range(len(input_list)):
        for j in range(len(input_list[i])):
            if len(out) <len(input_list[i]):
                out.append([])
            out[j].append(input_list[i][j])  
    return out
#transpose([
    #[1,2],
    #[3,4],
    #[5,6]
#])


# %%
#Create a function that can split a nested list into chunks
#chunk size is given by parameter
#return type: list
#function name must be: split_into_chunks
#input parameters: input_list,chunk_size

# %%
def split_into_chunks(input_list,chunk_size):
    out=[]
    c=0
    n=[]
    for l in input_list:        
        c=c+1
        n.append(l)
        if c==chunk_size:
            c=0
            out.append(n)
            n=[]
    if len(n)!=0:
        out.append(n)
    return out
#split_into_chunks([1,2,3,4,5,6,7,8], 3)

# %%
#Create a function that can merge n dictionaries
#return type: dictionary
#function name must be: merge_dicts
#input parameters: *dict

# %%
def merge_dicts(*dict):
    out={}
    for dic in dict:
        for key in dic.keys():
            out[key]=dic[key]
    return out
#merge_dicts({1:"one", 2: "two"}, {3:"three",4:"four",5:"five"})

# %%
#Create a function that receives a list of integers and sort them by parity
#and returns with a dictionary like this: {"even":[...],"odd":[...]}
#return type: dict
#function name must be: by_parity
#input parameters: input_list

# %%
def by_parity(input_list):
    out={}
    out["even"]=[]
    out["odd"]=[]
    for e in input_list:
        if e%2==0:
            out["even"].append(e)
        else:
            out["odd"].append(e)
    return out
#by_parity([1,2,3,4,5,6])

# %%
#Create a function that receives a dictionary like this: {"some_key":[1,2,3,4],"another_key":[1,2,3,4],....}
#and return a dictionary like this : {"some_key":mean_of_values,"another_key":mean_of_values,....}
#in short calculates the mean of the values key wise
#return type: dict
#function name must be: mean_key_value
#input parameters: input_dict

# %%
def mean_key_value(input_dict):
    for key in input_dict.keys():
        m=0
        for e in input_dict[key]:
            m+=e
        input_dict[key]=(m/len(input_dict[key]))
    return input_dict
#mean_key_value({"somekey":[1,2,3],"key": [5,5]})

# %%
#If all the functions are created convert this notebook into a .py file and push to your repo


