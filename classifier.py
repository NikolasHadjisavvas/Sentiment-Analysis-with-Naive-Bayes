import pandas as pd
import seaborn as sbs
import re
import csv
import nltk
import numpy as np
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# loading the datasets using pandas dataframe
dev_df = pd.read_csv("moviereviews/dev.tsv", sep='\t')
training_df = pd.read_csv("moviereviews/train.tsv", sep='\t')
test_df = pd.read_csv("moviereviews/test.tsv", sep='\t')

# Method taking the dataset as param and returning a
# dataframe with the corresponding 3 scale sentiments. 
def map_5_to_3(sentiments):
    sentiments_threeScale = {}
    for i in range(0,len(sentiments)):
        sentenceId = sentiments['SentenceId'][i]
        if sentiments['Sentiment'][i] == 1:
            sentiments_threeScale[sentenceId] = 0
        elif sentiments['Sentiment'][i] == 2:
            sentiments_threeScale[sentenceId] = 1
        elif (sentiments['Sentiment'][i] == 3) | (sentiments['Sentiment'][i] == 4):
            sentiments_threeScale[sentenceId] = 2  
        else:
            sentiments_threeScale[sentenceId] = sentiments['Sentiment'][i]
            
    phrases = list(sentiments['Phrase'])
    sentiments_threeScale_df = pd.DataFrame(sentiments_threeScale.items(), columns=['SentenceId', 'Sentiment'])
    sentiments_threeScale_df['Phrase'] = phrases
    return sentiments_threeScale_df


# User input for the choice of class numbers and feature selection
use_three_scale = input("Do you want to use three scale?(True or False):")
chosenModel = int(input("Choose model(1 for all words as features or 2 for only specific words(like adjectives)):"))

"""------------------------"""
# If three scale is chosen , then training data sentiments are modified by using the above method.
if use_three_scale == 'True':
    training_reviews_df = map_5_to_3(training_df)
else:
    training_reviews_df = training_df
"""------------------------"""

# Import NLTK's stopword list
from nltk.corpus import stopwords
sw = stopwords.words("english")
sw.remove('but')


# Takes dataframe as input, returns tokenised phrases(after stopword removal) along with the sentenceId
# in the form of a dictionary {sentenceId:[tokens]}
def pre_process(data):
    wordRE = re.compile(r'[A-Za-z!]+')
    processed={}
    for i in range(len(data['Phrase'])):
        phraseId = data['SentenceId'][i]
        tokens = wordRE.findall(data['Phrase'][i].lower())
        tokens = [x for x in tokens if x not in sw]
        processed[phraseId] = tokens
    return processed


# Returns the features of each phrase as a dict of form {sentenceID:features}
def feature_extraction(data, model=1):
    extr_feats = {}
    # Best  JJ , NN, RB
    acceptList = ['JJ','RB','RBR','RBS','JJR','JJS','VBD','VBG','VBN','VBP','VBZ','WDT']

    # Model 1: All words count as features.So it just returns the result of above method.
    if model == 1:
        extr_feats = pre_process(data)
        
    # Model 2: Select all tokens with POS tags in the acceptList(adjectives):
    if model == 2:
        processed_data = pre_process(data)
        for i in processed_data:
            text =processed_data[i]
            postags = nltk.pos_tag(text) # Get postags of tokens in the sentence.
            extr_feats[i] = [word for word, tag in postags if tag in acceptList] # Accept only those with the acceptable postags.
               
    return extr_feats

# Get the extracted features for later use.
extracted_features = feature_extraction(data = training_df, model=chosenModel)


# Returns all unique features, to be used when computing likelihoods of each feature to avoid 
# Examining the same feature twice.
def getAllFeatures(model=1):
    features = set()
    if model ==1:
        extr_features = feature_extraction(data = training_df, model = 1)
    elif model == 2:
        extr_features = feature_extraction(data =training_df, model = 2)
    for p in extr_features:
        features.update(extr_features[p])
    return features

# Get all distinct features for later use.
allFeatures = getAllFeatures(model = chosenModel)


# Method which uses the training data to return a dictionary of the form {distinct feature:{class:num of occurences in class}}
# Example, {good:{class 0: 19, class 2: 35}}
# This method is later used for the likelihood computation.
def index(three_scale = 'False'):
    index = {}
    for i in extracted_features: # For each sentences features
        for feature in extracted_features[i]: # Iterate over each feature in the sentence
            class_of_i = int(training_reviews_df[training_reviews_df["SentenceId"]==i]["Sentiment"]) # Get the class of the sentence
            if feature in index: #If feature already in index.
                if class_of_i in index[feature]: # If class of current sentence already in index[current feature].
                    index[feature][class_of_i] = index[feature][class_of_i]+1 # Just update the existing count.
                else: # If class not in index[current feature] then just initialize the count.
                    index[feature][class_of_i] = 1
            else: # if feature not already in index, then just initialize the nested dictionary and set the count to 1.
                index[feature] = {}
                index[feature][class_of_i] = 1
    return index

# Get the index
index = index(three_scale = use_three_scale)
print('INDEX FINISHED')

# Method taking a class as paramater and returning the amount of features(NOT UNIQUE) in that class.
# Later used for the likelihood computation.
def count_features_in_class(cl):
    count = 0
    # get all phrases in speciified class by indexing the training data.
    phrasesInClass = training_df[training_df["Sentiment"] == cl]["SentenceId"]
    # For each phrase ,add the number of features it contains to the counter.
    for k in phrasesInClass:
        count = count + len(extracted_features[k])
    return count

# Method returning the likelihoods of all features relative to all classes(according to scale used).
# Returns dictionary of the format {class: {feature: likelihood}}
def compute_likelihoods(three_scale='False'):

    likelihoods = {} # initialise likelihood dictionary
    voc = len(getAllFeatures(model = chosenModel)) # This is the number of distinct features in the training set(vocabulary).
    
    if three_scale == 'False':
        classes = [0,1,2,3,4]
    else:
        classes = [0,1,2]

    # For each class   
    for c in classes:
        allClassFeatures = count_features_in_class(c) # Use above method to get the number of features in class c.
        likelihoods[c] = {} # Initialise inner dictionary with key current class c.
        for f in allFeatures: # Iterate over each feature
            # If feature f occurs in class c then get the number of times it occurs, and then compute the likelihood
            # by using the likelihood formula.(here the normalised likelihood is used)
            if c in index[f]: 
                occurences = index[f][c]
                likelihoods[c][f] = (occurences + 1)/(allClassFeatures+ voc)

            # If feature f does not occur in class c then set the number of times it occurs to 0, and then compute the likelihood
            # by using the likelihood formula.
            else:
                occurences = 0
                likelihoods[c][f] = (occurences+ 1)/(allClassFeatures+voc)
    return likelihoods

likelihoods = compute_likelihoods(three_scale = use_three_scale) # Get likelihoods

# Method taking dataframe as input and returning the priors for all classes
# Depending if three scale or five scale is used.
def compute_priors(data,three_scale='False'):
    priors = {}
    if three_scale == 'False':
        classes = [0,1,2,3,4]
    else:
        classes = [0,1,2]
    for c in classes:
        count = len(data[data["Sentiment"]==c]) # All reviews with centiment i
        priors[c] = count/len(data)
    
    return priors


# Method taking a phrase/review upon classification and returning the posterior for every class
# depending on the scale used. 
# Output is of the type {class: posterior}
def compute_posterior(phrase, three_scale='False'):

    posteriors={}
    priors = compute_priors(training_reviews_df, three_scale = use_three_scale)
    
    if three_scale == 'False':
        classes = [0,1,2,3,4]
    else:
        classes = [0,1,2]

    for i in classes: #For each class.
        likelihood_product = 1 # Initialize the likelihood product as 1.
        for word in phrase: # For each word/feature in the phrase
            if word in likelihoods[i]: # If word is in the likelihoods data structure(appears in training data)
                likelihood_product = likelihood_product*likelihoods[i][word] # Update the likelihood product.
            else:
                # If word is not in the training data, instead of returning a posterior of 0, just
                # update the likelihood product by multiplying with 0.1 for the missing feature.
                likelihood_product = likelihood_product* 0.1
        
        # Store the posterior(prior of class * likelihood product) to the dictionary
        posteriors[i] = priors[i] * likelihood_product

    return posteriors

# Method taking data to be classified as parameter and returning the sentiment of each 
# review in the provided dataset.
def classification(reviews):
    predictions = {} #Initialise a predicitons dictionary
    processed = pre_process(reviews) # Apply the preprocess method to the data.
    for i in processed: # For each review in the set.
        posts = compute_posterior(processed[i], three_scale = use_three_scale) # Compute posteriors for the review.
        maxsKey = max([(v, k) for k, v in posts.items()],default=["Empty review provided"])[1] #Get the maximum posterior
        predictions[i] = maxsKey # The prediction is the max posterior.
    return predictions
classifier_predictions = classification(dev_df) # Apply the method to the development set.

# Code for printing the classification results to a tsv file.
label_file = "/tmp/output.tsv"
with open(label_file, 'w', encoding='utf8', newline='') as tsv_file:
    tsv_writer = csv.writer(tsv_file, delimiter='\t')
    tsv_writer.writerow(['SentenceId', 'Sentiment'])
    for i in classifier_predictions:
        tsv_writer.writerow([i, classifier_predictions[i]])


# Method classifying the input reviews by using the majority class.
# It will be used as a performance baseline.
def baseline(reviews):
    predictions = {}
    priors = compute_priors(training_reviews_df, three_scale=use_three_scale)
    majority_class = max([(v, k) for k, v in priors.items()])[1] #  Get the biggest prior(majority class)
    processed = pre_process(reviews)
    for i in processed:
        prediction = majority_class # Prediction is the majority class
        predictions[i] = prediction
    return predictions
# Get the baseline predictions.
baseline_predictions = baseline(dev_df)

# Get the right development set true sentiments according to the scaling system chosen by the user.
# Will later be used for getting the f1 scores and the confusion matrix. 
if use_three_scale == 'True':
    dev_trues = map_5_to_3(dev_df)[['SentenceId','Sentiment']]
else:
    dev_trues = dev_df[['SentenceId','Sentiment']]

# Converting the true values to a dictionary of the form {sentenceId: true sentiment}
trues = {}
for i in dev_trues['SentenceId']:
    s = int((dev_trues[dev_trues['SentenceId']==i])['Sentiment'])
    trues[i] = s


# Method for calculating confusion matrix, precision-recall-f1score for every class.
# Paramateres are: predictions obtained by the classification methos, observations which are the true sentiments
# of classified data.
def f1(predictions, observations, three_scale='False'):
    #Dimensions of the matrix according to scale used.
    if three_scale == 'True':
        dims = 3
    else: 
        dims = 5
    # Initialise the matrix with np.zeros()
    matrix = np.zeros([dims,dims])
    # Initialise precision,recall and f1 dictionaries
    precision={}
    recall={}
    f1score = {}
    #For each prediction
    for p in predictions:
        obs = observations[p] # Get the true sentiment of the prediction.
        if predictions[p] == obs: # If prediction is right then update the corresponding matrix cell.
            matrix[obs][obs] += 1
        elif predictions[p] != obs: # If prediction is wrong update the corresponding cell.
            matrix[predictions[p]][obs] += 1
            
    # Calculate precision,recall and f1score for all classes 
    # By indexing the confusion matrix.
    for i in range(0,dims):
        if sum(matrix[i,:]) != 0:
            precision[i] = matrix[i][i] / (sum(matrix[i,:]))
            recall[i] = matrix[i][i] / (sum(matrix[:,i]))
            # corresponding f1 is 2*precision[i]/precision[i]+recall[i].
            f1score[i] = 2* ((precision[i]*recall[i])/(precision[i]+recall[i])) 
        else:
            precision[i] = 0
            recall[i] = 0
            f1score[i] = 0
        
    
    return matrix,precision,recall,f1score


# Apply f1 method to get performance metrics.
conf_matrix,pre,rec,f1s = f1(classifier_predictions, trues, three_scale = use_three_scale)

# Print performance metrics.
print("Class 0 precision: ",pre[0], "\nClass 0 recall: ",rec[0],"\nClass 0 f1: ",f1s[0],'\n')
print("Class 1 precision: ",pre[1], "\nClass 1 recall: ",rec[1],"\nClass 1 f1: ",f1s[1],'\n')
print("Class 2 precision: ",pre[2], "\nClass 2 recall: ",rec[2],"\nClass 2 f1: ",f1s[2],'\n')
if use_three_scale == 'False':
    print("Class 3 precision: ",pre[3], "\nClass 3 recall: ",rec[3],"\nClass 3 f1: ",f1s[3],'\n')
    print("Class 4 precision: ",pre[4], "\nClass 4 recall: ",rec[4],"\nClass 4 f1: ",f1s[4],'\n')
    print("Macro F1-Score: ", sum(f1s.values())/5)
else:
    print("Macro F1-Score: ", sum(f1s.values())/3)
