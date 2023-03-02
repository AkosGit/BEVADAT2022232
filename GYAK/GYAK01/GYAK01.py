# %%
#Create a function that decides if a list contains any odd numbers.
#return type: bool
#function name must be: contains_odd
#input parameters: input_list

# %%
def contains_odd(input_list):
    for e in input_list:
        if e%2!=0:
            return True   
    return False  
#contains_odd([2,2,2])

# %%
#Create a function that accepts a list of integers, and returns a list of bool.
#The return list should be a "mask" and indicate whether the list element is odd or not.
#(return should look like this: [True,False,False,.....])
#return type: list
#function name must be: is_odd
#input parameters: input_list

# %%
def is_odd(input_list):
    oddnums=[]
    for e in input_list:
        if e%2==0:
            oddnums.append(False)
        else:
            oddnums.append(True)
    return oddnums
#is_odd([1,1,2,2])

# %%

#Create a function that accpects 2 lists of integers and returns their element wise sum. <br>
#(return should be a list)
#return type: list
#function name must be: element_wise_sum
#input parameters: input_list_1, input_list_2

# %%
def element_wise_sum(input_list_1, input_list_2):
    #decide which is smaller
    out=[]
    l=max(len(input_list_1), len(input_list_2))
    for i in range(l):
        num=0
        if i < len(input_list_1):
            num=num+input_list_1[i]
        if i < len(input_list_2):
            num=num+input_list_2[i]
        out.append(num)
    return out
#element_wise_sum([2,1],[2,2,2])

# %%
#Create a function that accepts a dictionary and returns its items as a list of tuples
#(return should look like this: [(key,value),(key,value),....])
#return type: list
#function name must be: dict_to_list
#input parameters: input_dict

# %%
def dict_to_list(input_dict):
    out=[]
    for key in input_dict.keys():
        out.append((key,input_dict[key]))
    return out
#dict_to_list({1:"one",2:"two"}) 

# %%
#If all the functions are created convert this notebook into a .py file and push to your repo


