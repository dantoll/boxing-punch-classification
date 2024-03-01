import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from standardizer import normate_dataset_period
import datetime as datetime
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction import MinimalFCParameters


class preprocessor():
    def __init__(self, filepath, data_stage='raw', raw_data_path="./data/complete_smartpunch_dataset_7606punches.json", random_state=42):
        
        self.fp = filepath
        
        self.labels = self.load_labels(raw_data_path)

        if data_stage=='raw':
            self.data = self.load_raw_data(self.fp)
            #split train and test data
            self.X_train, test_data, self.y_train, test_labels = train_test_split(self.data, self.labels, test_size=0.2, random_state=random_state)
            #split test and validation data
            self.X_test, self.X_val, self.y_test, self.y_val = train_test_split(test_data, test_labels, test_size=0.5, random_state=random_state)
            self.data_normalized = []
            self.features = []
                
        elif data_stage=='features':
            self.features = self.load_features(self.fp)
                        
        elif data_stage=='processed':
            self.data_normalized = self.load_processed_data(self.fp)
         
            self.features = []
        
        
            
        
        
        
    def pre_process(self, inter_extrapolation=None, dynamic_time_warping=None, split=None):
       
        if split != None:
            data, labels = self.get_split_data(split=split)
            
        elif len(self.data_normalized) != 0:
            data = self.get_normalized_data()
            labels = self.get_labels()
            
        else:
            data = self.get_raw_data()
            labels = self.get_labels()
            
        
        
        #Perform desired processing steps
        
        if inter_extrapolation != None:
            
            period_length = inter_extrapolation[0]
            sampling_rate = inter_extrapolation[1]
            data = self.inter_extrapolate(period_length, sampling_rate, data)
            
        if dynamic_time_warping != None:
            #NOT IMPLEMENTED HERE
            pass
        
                
        #ADD id values and concatenate the samples
        n_samples = len(data)
        for ID in range(n_samples):
            period_length = data[ID].shape[0]
            data[ID]['id'] = np.linspace(ID,ID, period_length).astype(int)
            
        concatenated = pd.concat(data)
        concatenated = concatenated.drop(['label', 'hand', 'annotator'], axis=1)
        
        if split != None:
            self.set_split_data_norm(concatenated, split)
        else:
            self.data_normalized = concatenated.copy()
        
        del data
        del concatenated
        

            
            
    def extract_features(self, split=None):
        
        if split != None:
            data, labels = self.get_split_data
        else:
            data = self.get_normalized_data()
            labels = self.get_labels()
            
        data = data.drop(['timestamp'], axis=1)
        
        if "Unnamed: 0" in data.keys():
            data = data.drop(['Unnamed: 0'], axis=1)
        
        settings = MinimalFCParameters()
        
        return extract_features(data, default_fc_parameters=settings, column_id = 'id')
            
        
            
                
                
                
    
    
    def encode_labels(self, label):
        if label == 'frontal':
            return 0
        if label == 'hook':
            return 1
        if label == 'upper-cut':
            return 2
        if label == 'no-action':
            return 3
            
    def inter_extrapolate(self, period_length, sampling_rate, data):
        """
        period_length in milliseconds
        sampling_rate in microseconds
        data raw
        """ 
        return normate_dataset_period(period_length, sampling_rate, data, interpolationKind='cubic')
            
    def load_features(self, filepath):
        with open(filepath) as fp:
            return pd.read_csv(fp)
    
    def load_labels(self, filepath):
        with open(filepath) as fp:
            dataset = pd.read_json(fp)
        
        y = dataset['label']
        y = y.apply(self.encode_labels)
        return y
            
    def load_raw_data(self, filepath):
        #Load json file and unpack into dataset format
        with open(filepath) as fp:
            return self.jsonData_to_dataset_in_timedifference_us(json.load(fp))
        
    def load_processed_data(self, filepath):
        #Load csv file
        with open(filepath) as fp:
            return pd.read_csv(fp)
        
                
    
            
    def jsonData_to_dataset_in_timedifference_us(self, data):
#***************************************************************************************/
#*    Title: jsonData_to_dataset_in_timedifference_us
#*    Author: Wagner, T
#*    Date: 2019
#*    Commit: 119c139
#*    Availability: https://github.com/smartpunch/timeseries_helpers/blob/master/database_importer.py
#*
#***************************************************************************************/

        
        """
        Creates a list of dataframe objects from a given json object. Converts the timestamp col with absolute timestamps in us.
        The last timestamp is the period time in us since the punch started.
        
        Keyword arguments:
        data            -- JSON Database representation dumped from mongoDB with timestamps in nanoseconds (type: object)
        
        Returns:
        list            -- List object containing the datasets as dataframe objects with timestamps in 'since last timestamp' format.
        """

        the_cols = ['x', 'y', 'z', 'timestamp', 'label', 'hand', 'annotator']
        the_data = []

        for value in data:
            the_raws = []
            the_indxs = []
            idx = 0
            raw_time_us = 0
            for raw in value['raws']:
                raw_time_us += int(raw['timestamp'])/1000
                the_raws.append([raw['x'], raw['y'], raw['z'], int(
                raw_time_us), value['label'], value['hand'], value['annotator']])
                the_indxs.append(idx)
                idx += 1
            the_data.append(pd.DataFrame(the_raws, the_indxs, the_cols))                
        return the_data
    
    
    
    #================
    #               
    #GET            
    
    
    
    def get_normalized_data(self):
        return self.data_normalized.copy()
    
    def get_raw_data(self):
        return self.data.copy()
    
    def get_features(self):
        return self.features.copy()
    
    def get_labels(self):
        return self.labels.copy()
    
    def get_split_data(self, split):
        if split=='train':
            return self.X_train_norm.copy(), self.y_train.copy()
        
        elif split=='test':
            return self.X_test_norm.copy(), self.y_test.copy()
            
        elif split=='validation':
            return self.X_val_norm.copy(), self.y_val.copy()
        
        
    def get_split_labels(self, split):
        if split=='train':
            return self.y_train.copy()
        
        elif split=='test':
            return self.y_test.copy()
            
        elif split=='validation':
            return self.y_val.copy()
        
        elif split=='all':
            return self.y_train.copy(), self.y_test.copy(), self.y_val.copy()
        
        
    #================
    #               
    #SET
    
    def set_split_data_norm(self, data, split):
        if split=='train':
            self.X_train_norm = data.copy()
        
        elif split=='test':
            self.X_test_norm = data.copy()
            
        elif split=='validation':
            self.X_val_norm = data.copy()
            
        
    
            
        
        
    
        
        
        
