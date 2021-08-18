#!/usr/bin/env python
# coding: utf-8


#Script to obtain data 

#import all the relevant libraries
import spotipy
import time
from IPython.core.display import clear_output
from spotipy import SpotifyClientCredentials, util


#spotify authentication credentials
client_id='e8e1c5445d644b86a7f6828462c402a0'
client_secret='be507d0c358f4b8ca1fdcf419ef0f527'
redirect_uri='http://localhost:8080/'
username = 'akshatsharma6301'
scope = 'playlist-modify-public'



#Credentials to access the Spotify Music Data
manager = SpotifyClientCredentials(client_id,client_secret)
sp = spotipy.Spotify(client_credentials_manager=manager)

#Credentials to access to the Spotify User's Playlist, Favorite Songs, etc. 
token = util.prompt_for_user_token(username,scope,client_id,client_secret,redirect_uri) 
spt = spotipy.Spotify(auth=token)


# funtion to get albums using albumID

def get_albums_id(ids):
    album_ids = []
    results = sp.artist_albums(ids)
    for album in results['items']:
        album_ids.append(album['id'])
    return album_ids


# function to get list of songs from an album

def get_album_songs_id(ids):
    song_ids = []
    results = sp.album_tracks(ids,offset=0)
    for songs in results['items']:
        song_ids.append(songs['id'])
    return song_ids


# funciton to get song features used for training and prediction

def get_songs_features(ids):

    meta = sp.track(ids)
    features = sp.audio_features(ids)

    # meta
    name = meta['name']
    album = meta['album']['name']
    artist = meta['album']['artists'][0]['name']
    release_date = meta['album']['release_date']
    length = meta['duration_ms']
    popularity = meta['popularity']
    ids =  meta['id']

    # features
    acousticness = features[0]['acousticness']
    danceability = features[0]['danceability']
    energy = features[0]['energy']
    instrumentalness = features[0]['instrumentalness']
    liveness = features[0]['liveness']
    valence = features[0]['valence']
    loudness = features[0]['loudness']
    speechiness = features[0]['speechiness']
    tempo = features[0]['tempo']
    key = features[0]['key']
    time_signature = features[0]['time_signature']

    track = [name, album, artist, ids, release_date, popularity, length, danceability, acousticness,
            energy, instrumentalness, liveness, valence, loudness, speechiness, tempo, key, time_signature]
    columns = ['name','album','artist','id','release_date','popularity','length','danceability','acousticness','energy','instrumentalness',
                'liveness','valence','loudness','speechiness','tempo','key','time_signature']
    return track,columns


# function to get spotify songID and ArtistID from PlaylistID

def get_songs_artist_ids_playlist(ids):
    playlist = sp.playlist_tracks(ids)
    songs_id = []
    artists_id = []
    for result in playlist['items']:
        songs_id.append(result['track']['id'])
        for artist in result['track']['artists']:
            artists_id.append(artist['id'])
    return songs_id,artists_id


# function to get albums

def download_albums(music_id,artist=False):
    
    if artist == True:
        ids_album = get_albums_id(music_id)
    else:
        if type(music_id) == list:
            ids_album = music_id
        elif type(music_id) == str:
            ids_album = list([music_id])

    tracks = []
    for ids in ids_album:
        #Obtain IDs of songs in album 
        song_ids = get_album_songs_id(ids=ids)
        #Get features of songs in album
        ids2 = song_ids
        
        print(f"Album Length: {len(song_ids)}")
         
        time.sleep(.6)   
        track, columns = get_songs_features(ids2)
        tracks.append(track)

        print(f"Song Added: {track[0]} By {track[2]} from the album {track[1]}")
        clear_output(wait = True)
        
    clear_output(wait = True)
    print("Music Downloaded!")
 
    return tracks,columns


# function to get songs and their features from a playlist

def download_playlist(id_playlist,n_songs):
    songs_id = []
    tracks = []

    for i in range(0,n_songs,100):
        playlist = spt.playlist_tracks(id_playlist,limit=100,offset=i)
        
        for songs in playlist['items']:
            songs_id.append(songs['track']['id'])
    
    counter = 1
    for ids in songs_id:
        
        time.sleep(.6)
        track,columns = get_songs_features(ids)
        tracks.append(track)

        print(f"Song {counter} Added:")
        print(f"{track[0]} By {track[2]} from the album {track[1]}")
        clear_output(wait = True)
        counter+=1
    
    clear_output(wait = True)
    print("Music Downloaded!")

    return tracks,columns




import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sys,json

#Libraries to create the multiclass model
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
import keras
from keras import models
from keras import layers
#Import tensorflow and disable the v2 behavior and eager mode
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

#Library to validate the model
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

# Additional Classifiers for future use if required
#from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.ensemble import RandomForestRegressor
#import xgboost as xgb

    



def preprocessing(df):
    col_features = df.columns[6:-3]
    X= MinMaxScaler().fit_transform(df[col_features])
    X2 = np.array(df[col_features])
    Y = df['mood']
    #Encodethe categories
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_y = encoder.transform(Y)
    target = pd.DataFrame({'mood':df['mood'].tolist(),'encode':encoded_y}).drop_duplicates().sort_values(['encode'],ascending=True)
 
    return X,encoded_y,target,X2




def trainModel(X,encoded_y):
    X_train,X_test,Y_train,Y_test = train_test_split(X,encoded_y,test_size=0.2,random_state=15)
    estimator = KerasClassifier(build_fn=base_model,epochs=800,batch_size=200,verbose=0)
    estimator.fit(X_train,Y_train)
    y_preds = estimator.predict(X_test)
    acc = accuracy_score(Y_test,y_preds)
    print(f'Successfully trained model with an accuracy of {acc:.2f}')
    return estimator


#Base Model

def base_model():
    #Create the model
    model = Sequential()
    #Add 1 layer with 11 nodes,input of 10 dim with relu function
    model.add(Dense(11,input_dim=10,activation='relu'))
    #Add 1 layer with output 4 and softmax function
    model.add(Dense(4,activation='softmax'))
    #Compile the model using sigmoid loss function and adam optim
    model.compile(loss='categorical_crossentropy',optimizer='adam',
                 metrics=['accuracy'])
    return model

'''
def base_model2():
    model2 = models.Sequential()
    model2.add(layers.Dense(256, activation='relu', input_shape=(10,)))

    model2.add(layers.Dense(128, activation='relu'))

    model2.add(layers.Dense(64, activation='relu'))
    
    model2.add(layers.Dense(32, activation='relu'))
    
    model2.add(layers.Dense(16, activation='relu'))

    model2.add(layers.Dense(4, activation='softmax'))

    model2.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model2
'''



#Configure the model with ML algorithms for comparison in future
#estimator2 = RandomForestClassifier(n_estimators=500)
#estimator3 = xgb.XGBClassifier(max_depth=3, n_estimators=500, learning_rate=0.05)
# estimator4 = KNeighborsClassifier()
# estimator5 = GaussianNB()
# estimator6 = SVC(C=1, gamma=0.1)
# estimator7 = DecisionTreeClassifier()




#Evaluate the model using KFold cross validation
#kfold = KFold(n_splits=5,shuffle=True)
#results = cross_val_score(estimator,X,encoded_y,cv=kfold)
#print(results.max())
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100,results.std()*100))

'''cm = confusion_matrix(Y_test,y_preds)
ax = plt.subplot()
sns.heatmap(cm,annot=True,ax=ax)

labels = target['mood']
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)'''
#plt.show()



def predict_mood(id_song,target,X2):
    #Join the model and the scaler in a Pipeline
    pip = Pipeline([('minmaxscaler',MinMaxScaler()),('keras',KerasClassifier(build_fn=base_model,epochs=300,
                                                                             batch_size=100,verbose=0))])
    #Fit the Pipeline
    pip.fit(X2,encoded_y)

    #Obtain the features of the song
    preds = get_songs_features(id_song)
    #Pre-process the features to input the Model
    preds_features = np.array(preds[0][6:-2]).reshape(-1,1).T

    #Predict the features of the song
    results = pip.predict(preds_features)

    mood = np.array(target['mood'][target['encode']==int(results)])
    name_song = preds[0][0]
    artist = preds[0][2]

    return "{0} by {1} is a {2} song".format(name_song,artist,mood[0].upper())
    #print(f"{name_song} by {artist} is a {mood[0].upper()} song")



def getTrackId(trackName,artistName):
    artist = artistName
    title = trackName
    results = sp.search(q='artist:' + artist + ' track:' + title, type='track')
    #print(results)
    trackId = results['tracks']['items'][0]['id']
    print(trackId)
    return trackId


if __name__ == '__main__':
    df = pd.read_csv("/home/akshat/Downloads/data_moods.csv")
    X,encoded_y,target,X2 = preprocessing(df)
    Classifier = trainModel(X,encoded_y)
    trackName = sys.argv[1]
    artistName =  sys.argv[2]
    print(trackName)
    print(artistName)
    trackId = getTrackId(trackName,artistName)
    result = predict_mood(trackId,target,X2)
    print (result)
    tmp = {'prediction': result}
    print (json.dumps(tmp))


