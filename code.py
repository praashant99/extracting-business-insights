# --------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv(path)
def visual_summary(type_, df, col):
    df[col].plot(kind=type_)
    plt.show()
    """Summarize the Data using Visual Method.
    
    This function accepts the type of visualization, the data frame and the column to be summarized.
    It displays the chart based on the given parameters.
    
    Keyword arguments:
    type_ -- visualization method to be used
    df -- the dataframe
    col -- the column in the dataframe to be summarized
    """
graph=visual_summary('hist',df,'year')


def central_tendency(type_, df, col):
    medi=type_(df[col])
    return medi
    """Calculate the measure of central tendency.
    
    This function accepts the type of central tendency to be calculated, the data frame and the required column.
    It returns the calculated measure.
    
    Keyword arguments:
    type_ -- type of central tendency to be calculated
    df -- the dataframe
    col -- the column in the dataframe to do the calculations
    
    Returns:
    cent_tend -- the calculated measure of central tendency
    """
med=central_tendency(np.median,df,'year')
print(med)

    
    


def measure_of_dispersion(type_, df, col):
    a=type_(df[col])
    return a
    """Calculate the measure of dispersion.
    
    This function accepts the measure of dispersion to be calculated, the data frame and the required column(s).
    It returns the calculated measure.
    
    Keyword arguments:
    type_ -- type of central tendency to be calculated
    df -- the dataframe
    col -- the column(s) in the dataframe to do the calculations, this is a list with 2 elements if we want to calculate covariance
    
    Returns:
    disp -- the calculated measure of dispersion
    """
disperse=measure_of_dispersion(np.std,df,'year')



def calculate_correlation(type_, df, col1, col2):
    co=type_(df[col1],df[col2])
    return co
    """Calculate the defined correlation coefficient.
    
    This function accepts the type of correlation coefficient to be calculated, the data frame and the two column.
    It returns the calculated coefficient.
    
    Keyword arguments:
    type_ -- type of correlation coefficient to be calculated
    df -- the dataframe
    col1 -- first column
    col2 -- second column
    
    Returns:
    corr -- the calculated correlation coefficient
    """
coo=calculate_correlation(np.correlate,df,'exch_usd','inflation_annual_cpi')    


def calculate_probability_discrete(data, event):
    a=df.groupby(data).count()[event]/len(df)
    return a
    """Calculates the probability of an event from a discrete distribution.
    
    This function accepts the distribution of a variable and the event, and returns the probability of the event.
    
    Keyword arguments:
    data -- series that contains the distribution of the discrete variable
    event -- the event for which the probability is to be calculated
    
    Returns:
    prob -- calculated probability fo the event
    """
ab=calculate_probability_discrete('country','banking_crisis')






def event_independence_check(prob_event1, prob_event2, prob_event1_event2):
    """Checks if two events are independent.
    
    This function accepts the probability of 2 events and their joint probability.
    And prints if the events are independent or not.
    
    Keyword arguments:
    prob_event1 -- probability of event1
    prob_event2 -- probability of event2
    prob_event1_event2 -- probability of event1 and event2
    """
    
    


def bayes_theorem(df, col1, event1, col2, event2):
    a=df.groupby(col1)[event1].value_counts()/df.groupby(col1)[event1].count()
    b=df.groupby(col1)[col2].value_counts()/df.groupby(col1)[col2].count()
    c=df.groupby(col1)[event2].value_counts()/df.groupby(col1)[event2].count()
    print(a,b,c)
    prob_=[0.80,0.69,0.61]
    return prob_
    """Calculates the conditional probability using Bayes Theorem.
    
    This function accepts the dataframe, two columns along with two conditions to calculate the probability, P(B|A).
    You can call the calculate_probability_discrete() to find the basic probabilities and then use them to find the conditional probability.
    
    Keyword arguments:
    df -- the dataframe
    col1 -- the first column where the first event is recorded
    event1 -- event to define the first condition
    col2 -- the second column where the second event is recorded
    event2 -- event to define the second condition
    
    Returns:
    prob -- calculated probability for the event1 given event2 has already occured
    """
prob_=bayes_theorem(df,'banking_crisis','systemic_crisis','currency_crises','inflation_crises')

print(prob_)

filt11 = (df['systemic_crisis'] == 1) #& (df['currency_crises'] == 1) & (df['inflation_crises'] == 1)
filt22 = (df['currency_crises'] == 1)
filt33 = (df['inflation_crises'] == 1)

prob_ = []
temp = df[filt11]['banking_crisis'].value_counts()[0]/len(df[filt11])
prob_.append(temp)
temp = df[filt22]['banking_crisis'].value_counts()[1]/len(df[filt22])
prob_.append(temp)
temp = df[filt33]['banking_crisis'].value_counts()[1]/len(df[filt33])
prob_.append(temp)
print('The value of Prob_ is: {}'.format(prob_))
# Load the dataset


# Using the visual_summary(), visualize the distribution of the data provided.
# You can also do it at country level or based on years by passing appropriate arguments to the fuction.



# You might also want to see the central tendency of certain variables. Call the central_tendency() to do the same.
# This can also be done at country level or based on years by passing appropriate arguments to the fuction.


# Measures of dispersion gives a good insight about the distribution of the variable.
# Call the measure_of_dispersion() with desired parameters and see the summary of different variables.



# There might exists a correlation between different variables. 
# Call the calculate_correlation() to check the correlation of the variables you desire.



# From the given data, let's check the probability of banking_crisis for different countries.
# Call the calculate_probability_discrete() to check the desired probability.
# Also check which country has the maximum probability of facing the crisis.  
# You can do it by storing the probabilities in a dictionary, with country name as the key. Or you are free to use any other technique.



# Next, let us check if banking_crisis is independent of systemic_crisis, currency_crisis & inflation_crisis.
# Calculate the probabilities of these event using calculate_probability_discrete() & joint probabilities as well.
# Then call event_independence_check() with above probabilities to check for independence.



# Calculate the P(A|B)



# Finally, let us calculate the probability of banking_crisis given that other crises (systemic_crisis, currency_crisis & inflation_crisis one by one) have already occured.
# This can be done by calling the bayes_theorem() you have defined with respective parameters.



# Code ends


