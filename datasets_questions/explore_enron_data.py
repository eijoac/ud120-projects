#!/usr/bin/python

""" 
    starter code for exploring the Enron dataset (emails + finances) 
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import pprint

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# print enron_data
print "number of persons: %d" %len(enron_data)
print "number of features: %d" %len(enron_data['METTS MARK'])

num_poi = 0
for key in enron_data:
    if enron_data[key]['poi']:
        num_poi = num_poi + 1

print "number of poi: %d" %num_poi

pp = pprint.PrettyPrinter(indent=4)

print "James Prentice\n"
pp.pprint(enron_data['PRENTICE JAMES'])

print "Wesley Colwell\n"
pp.pprint(enron_data['COLWELL WESLEY'])

print "Jeffrey Skilling"
pp.pprint(enron_data['SKILLING JEFFREY K'])

print "Kenneth Lay - total payments", \
    enron_data['LAY KENNETH L']['total_payments']

print "Jeffrey Skilling - total payments", \
    enron_data['SKILLING JEFFREY K']['total_payments']

print "Andrew Fastow - total payments", \
    enron_data['FASTOW ANDREW S']['total_payments']
    
num_salary = 0
num_email_add = 0
num_total_payment_nan = 0
num_total_payment_poi_nan = 0

for key in enron_data:
    if enron_data[key]['salary'] <> 'NaN':
        num_salary = num_salary + 1
        
    if enron_data[key]['email_address'] <> 'NaN':
        num_email_add = num_email_add + 1
        
    if enron_data[key]['total_payments'] == 'NaN':
        num_total_payment_nan = num_total_payment_nan + 1
        
    if enron_data[key]['poi'] == True \
        and enron_data[key]['total_payments'] == 'NaN':
        num_total_payment_poi_nan = num_total_payment_poi_nan + 1
    
print "number of persons that have salary data: %d" %num_salary
print "number of persons that have known email address: %d" %num_email_add

percentage_total_payment_nan = num_total_payment_nan/float(len(enron_data))
print "number of persons that have total payments as NaN: %d" \
    %num_total_payment_nan
print "percentage persons that have total payments as NaN: %.2f" \
     %percentage_total_payment_nan

percentage_total_payment_poi_nan = num_total_payment_poi_nan/float(num_poi)
print "percentage poi that have total payments as NaN: %.2f"\
     %percentage_total_payment_poi_nan
     
exercised_stock_options_list = []
for key in enron_data:
    exercised_stock_options_list.append(enron_data[key]['exercised_stock_options'])
print sorted(exercised_stock_options_list)
    


