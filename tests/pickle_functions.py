import pickle 
  
def pickle_var(var, filename:str):
    with open(filename+'.pkl', 'wb') as file: 
        pickle.dump(var, file) 
        
    return

def unpickle_var(filename:str):
    with open(filename+'.pkl', 'rb') as file: 
        myvar = pickle.load(file) 
        return myvar
    return 