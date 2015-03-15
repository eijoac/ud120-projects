#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    
    cleaned_data = []
    
        ### your code goes here
    error = abs(net_worths - predictions)
        
    data = zip(ages, net_worths, error)
        
    data.sort(key = lambda tup: tup[2])
    
    end = int(0.9 * len(data))
    
    cleaned_data = data[0 : end]
    
    print "cleaned data length: ", len(cleaned_data)
    
    
    
    
    
    return cleaned_data

