#importing libraries
import pandas as pd
import numpy as np
from ML_pipeline import utils
import re
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

#NLP DATA PREPROCESSING AND FEATURE EXTRACTION HELPER FUNCTIONS
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')

#HELPER FUNCTIONS TO PREPARE THE DATA

# FUNCTION TO DROP NULL VALUES FROM THE DATAFRAME ASLO KNOWN AS THE DF
def drop_null(df):
    try:
        data=df.dropna().reset_index(drop=True)
    except Exception as e:
        print(e)
    else:
        return data
    

    #FUNCTION TO DROP COLUMN
    def drop_columns(df, collist):
        try:
            df.drop(collist, axist=1, inplace=True)
        except Exception as e:
            print(e)

        else:
            return df


#FUNCTION TO STRIP  DATA FROM A COL AND SAVE IN NEW COL
def strip_data(df, col, newcol, idx):
    '''function to strip data from an object column and save it in a new columnof the dataframe
    df: df is the dataframe to be processed
    col: column on which stripping `has to e done
    newcol: name of the new column
    idx: stripping index for the value to be stripped
    '''
    try:
        df[newcol]=df[col].apply(lambda x: x.split(',')[idx].lower().strip())   
    except Exception as e:
        print(e)
    else:
        return df
    
#FUNCTION TO SEPERATE NUMBERS OUT
def strip_number(df, col):
    '''the function strips the numerical data from a column
    df:dataframe
    col: column'''
    try:
        newcol=col + 'cleaned'
        numbers= re.compile(r"[-+]?(\d+*\.\d+|\d+)")
        df[newcol]= df[col].apply(lambda x: numbers.findall(x)[0] if len(numbers.findall(x))> 0 else 0)
    except Exception as e:
        print (e)

    else:
        return df
    

#FUNCTION TO CLEAN CATEGORICAL OR TEXTUAL DATA CASE INCONSISTENCIES
def clean_text(df,col):
    '''the function removes case inconsistences from categories/textual data
    df: dataframe
    col: column
    '''
    try:
        newcol=col + 'cleaned'
        df[newcol] = df[col].apply(lambda x: x.lower().strip())
    except Exception as e:
        print (e)
    else:
        return df
    
def clean_numbers(df, col):
    ''' the function creates binary encoded features for categorical data
    df: dataframe
    col:column
    mapdict: mapping dictionary
    '''
    try:
        newcol=col +'cleaned'
        numbers= re.compile(r"[-+]?(\d+*\.\d+|\d+)")
        df[newcol]= (df[col].apply(lambda x: np.float(numbers.findall(str(x)))[0])
                                                       if len(numbers.findall(str()))>0 else np.nan)
    except Exception as e:
        print (e)
    else:
        return df
    

#FUNCTION TO ENCODE BINARY FEATURES
def binary_encoder(df, col, mapdict):
    '''the function creates binary encoded featuresfor categorical data 
    df: datafram
    col; column
    mapdict: mapping dictionary
    '''
    try:
        newcol= col + "cleaned"
        df[newcol]= (df[col].apply(lambda x: x.lower().strip()).map(mapdict))
    except Exception as e:
        print(e)
    else:
        return df
    

#FUNCTION TO CALCULATE AVERAGE PROPERTY AREA (FEATURE CLEANING)
def avg_area_calculation(df, col):
    '''the function calcullates the average area of the property by first stripping out the numbers from the area ranges provide
    df: dataframe
    col;column
    '''

    def avg_property_area(x):
    #find numbers from the pattern
      try:
        numbers= re.compile(r"[-+]?(\d+*\.\d+|\d+)")
        x= numbers.findall(x)
        # if a single number is given, return that as area
        if len(x) ==1:
            return np.float(x[0])
        
        #if a range of numbers is given, calcultate the average and return that as area
        elif len(x) ==2:
              return (np.float(x[0])+np.float(x[1]))/2
        else:
            return -99
      except Exception as e:
        print (e)

      try:
          newcol=col + 'cleaned'
          df[newcol]= df[col].apply(lambda x: avg_property_area(str(x)))
      except Exception as e:
          print (e)
      else:
          return df
      
#FUNCTION TO TREAT THE OUTLIERS
def outlier_treatment (df, cols_to_treat):
    #outlier treatment
    def clip_outliers(df, col):
        try:
            q_l = df[col].min()
            q_h = df[col].quantile(0.95)
            df[col]= df[col].clip(lower = q_l, upper= q_h)
            return df
        except Exception as e:
            print (e)
    try:
        for col in cols_to_treat:
            df= clip_outliers(df, col)
    except Exception as e:
        print(e)
    else:
        return df
    

#FUNCTION TO CALCULATE ROW WISE SUM OF COLUMNS
def sum_of_cols(df, col_list, newcol):
    try:
        temp =df[col_list]
        temp[newcol]=temp.sum(axis=1)
        df[newcol]= temp[newcol]
    except Exception as e:
        print(e)
    else:
        return df
        
#TEXT CLEANING
#PREPROCESSIN THE TEXT DATA
REPLACE_BY_SPACE_RE= re.compile("[/(){}\[\]\|@,;!]")
BAD_SYMBOLS_RE= re.compile("[^0-9a-z #+_]")
STOPWORDS_nlp= set(stopwords.words('english'))

#custom stoplist
stoplist=["i","project","living","home",'apartment',"pune","me","my","myself","we","our","ours","ourselves","you","you're","you've","you'll","you'd","your",
            "yours","yourself","yourselves","he","him","his","himself","she","she's","her","hers","herself","it",
            "it's","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","that'll",
            "these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did",
            "doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about",
            "against","between","into","through","during","before","after","above","below","to","from","up","down","in","out",
            "on","off","over","under","again","further","then","once","here","there","when","where","why","all","any",
            "both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too",
            "very","s","t","can","will","just","don","don't","should","should've","now","d","ll","m","o","re","ve","y","ain",
            "aren","couldn","didn","doesn","hadn","hasn",
            "haven","isn","ma","mightn","mustn","needn","shan","shan't",
            "shouldn","wasn","weren","won","rt","rt","qt","for",
            "the","with","in","of","and","its","it","this","i","have","has","would","could","you","a","an",
            "be","am","can","edushopper","will","to","on","is","by","ive","im","your","we","are","at","as","any","ebay","thank","hello","know",
            "need","want","look","hi","sorry","http", "https","body","dear","hello","hi","thanks","sir","tomorrow","sent","send","see","there","welcome","what","well","us"]

STOPWORDS_nlp.update(stoplist)













STOPWORDS_nlp.update(stoplist)

#FUNCTION TO PREPROCESS THE TEXT  KEY FACTOR
def text_prepare(text):
    '''
    text: a string
    return: modified initial string
    
    '''
    try:
        text= text.replace("\d+","") # this goes for removing a digit
        text= re.sub(r"(?:\@https\://)\s+","", text) # removing mentions and urls
        text= text.lower() # lowercase text
        text= re.sub("[0-9]+", "", text)
        text= REPLACE_BY_SPACE_RE.sub("", text)  # replace REPLACE_BY_SPACE_RE symbols by space text
        text= BAD_SYMBOLS_RE.sub("", text) #delete BAD_SYMBOLS_RE from the text
        text= ''.join([word for word in text.split() if word not in STOPWORDS_nlp]) #DELETE STOPWORDS FROM TEXT
        text= text.strip()
        return text
    except Exception as e:
        print(e)


#POS COUNTER
def pos_counter(x,pos):
    """
    Returns the count for the given parts of speech tag
    
    NN- Noun
    VB- Verb
    JJ- Adjective
    RB- Adverb
    """
    try:
        tokens= nltk.word_tokenize(x.lower())
        tokens= [word for word in tokens if word not in STOPWORDS_nlp]
        text= nltk.Text(tokens)
        tags= nltk.pos_tag(text)
        counts= Counter(tag for word, tag in tags)
        return counts[pos]
    except Exception as e:
        print(e)


#FUNCTION TO COUNT VECTORIZE TEXT DATA AND COMBINE IT TO THE ORIGINAL DATAFRAMEUSING ngrams

def count_vectorize(df, textcol, ngrams, max_features):
    """df: dataframe
    textcool; text column to be vectorized 
    ngrams: ngram_range e.g (2,2)
    max_features: number of maximum most frequent features"""
    try:
        cv= CountVectorizer(ngram_range=ngrams, max_features=max_features)
        #cv = countvectorizer()
        cv_object= cv.fit(df[textcol])
        X = cv_object.transform(df[textcol])
        df_ngram = pd.DataFrame(X.toarray(), columns=cv_object.get_feature_names())

        #ADDING THIS TO THE MAIN DATAFRAME
        df_final =pd.concat([df.reset_index(drop=True), df_ngram.reset_index(drop=True)], axis=1)
    except Exception as e:
        print (e)
    else:
        return df_final, cv_object
    

#FUNCTION COMBINE THE PREPROCESSING STEPS FOR THE PROBLEM STATEMENT 

def preprocess_data(df):
    """
    the function returns a clean and preprocessed dataset ready to usefor training purpose
    input:
    --df: Raw data dataframe
    
    output:
    --df1: processed Dataframe"""

    try:
        #preprocessing data
        #stripping location details
        for idx, col in enumerate (["city","state","country"]):
            df = strip_data(df,"location ", col, idx)

        #strip numbers for property type
        df= strip_number(df,"property type")

        #cleaning text columns
        for col in ["sub-area","Company Name","Township Name/ Society Name","Description"]:
            df = clean_text(df, col)


        #cleaning and encoding binary features
        for col in ["ClubHouse","school/ University in Township ","hospital in Township","Mall IN Township","park/ jogging track","Swimming pool","gym"]:
            df = binary_encoder(df, col, {"yes":1, "no":0})

        
        #cleaning numerical features: Avg area
        df = avg_area_calculation(df,"property area in takoradi. Ft.")


        #dropping null values
        df= drop_null(df)
        df= clean_numbers(df, "price in lakhs")

        #dropping unneccesary columns
        features = df.columns.tolist()[18:]
        df1= df[features]



        #treating outliers in the numeric columns
        cols_to_treat = ["Property area in takoradi. ft. Cleaned","price in vee estate Cleaned"]
        df1= outlier_treatment(df1, cols_to_treat)








        #feature engineering and extraction

        #saving the mapping dict for inference use
        sub_area_price_map= df1.groupby("sub-Area cleaned")["Price in VEE estate cleaned"].mean().to_dict()
        utils.pickle_dump(sub_area_price_map, "outpput/sub_area_price_map.pkl")

        #creating the price by sub-area feature
        df1["Price by sub-area"] = df1.groupby("Sub-area cleaned")["Price in VEE estatecleaned"].transform("mean")


        #amenities col
        amenities_col= df1.columns.tolist()[8:15]
        df1 = sum_of_cols(df1, amenities_col, "Amenities score")


        #saving the mapping dict for interence use
        amenities_score_price_map = df1.groupby("Amenities score")["Price in VEE estate cleaned"].mean().to_dict()
        utils.pickle_dump(amenities_score_price_map, "output/amenities_score_price_map.pk1")

        #creating the price by amenities score feature
        df1["Price by Amenities score"]= df1.groupby("Amenities score")["Price in VEE estate cleaned"].transform("mean")



        #cleaning the description column and creating pos features 
        df1["Description Cleaned"]= df1["Description cleaned"].astype(str).apply(text_prepare)
        df1["Noun_counts"]= df1["Description cleaned"].apply(lambda x: pos_counter(x, "NN"))
        df1["Verb_counts"]= df1["Description cleaned"].apply(lambda x: pos_counter(x, "VB"))+ pos_counter ( "RB")
        df1["Adjective_counts"]= df1["Description cleaned"].apply(lambda x: pos_counter(x, "JJ"))


        #creating count vectors
        df1, cv_object = count_vectorize(df1, "Description cleaned", (2,2), 10)


        #dump cv_object for interference purpose
        utils.pickle_dump(cv_object,"output/count_vectorizer.pk1")

        #final df
        #selecting only numerical features
        cols_to_drop = ["City","State","Country","Sub-Area cleaned","Township Name/Society Name cleaned",
                        "Description Cleaned", "Company Name Cleaned"]
        df1= cols_to_drop(df1, cols_to_drop)



        #change feature names, dump the object for interference purpose
        features = list(df1.columns)

        featuresMod= ["PropertyType",
                      "Clubhouse"
                      'School-University-in_Township',
                      "Hospital_in_Township",
                      "Mall_in-Township",
                      "Park_Jogging_track",
                      "Swimming_Pool",
                      "Gym",
                      "Property_Area_in_Takoradift",
                      "Price_in_VEE_estate",
                      "Price_by_sub_area",
                      "Amenitiees_score",
                      "Price_by_amenities_score",
                      "Noun_Counts",
                      'Adjective_Counts',
                      "Verb_Counts",
                      "Boast_Elegent",
                      "Elegant_Towers",
                      "every_day",
                      "great_community",
                      "mantra_gold",
                      "offering_bedroom",
                      "quality_specification",
                      "stories_offering",
                      "towers_stories",
                      "world_class"]
        #dump object for interference purposes
        utils.pickle_dump(dict(zip(features, featuresMod)), "output/raw_features_mapping.pk1")
        utils.pickle_dump(featuresMod, "output/ features.pk1")
        df1.columns = featuresMod
    
    except Exception as e:
        print(e)
    else:
        return df1

        





 

