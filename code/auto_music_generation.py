#!/usr/bin/env python
# coding: utf-8

# In[2]:


from music21 import *


# In[233]:


import glob
import tqdm
import tensorflow as tf
from tensorflow.keras.layers import LSTM,Dropout,Dense
from tensorflow.keras.models import Sequential,Model,load_model
from sklearn.model_selection import train_test_split
import numpy as np
import random


# In[15]:


def read_files(file):
    notes=[]
    notes_to_parse=None
    try:
        midi=converter.parse(file)
        instrmt=instrument.partitionByInstrument(midi)
        print(file,"---",midi)
        for part in instrmt.parts:
            if 'Piano' in str(part):
                notes_to_parse=part.recurse()
                for element in notes_to_parse:
                     if type(element)==note.Note:
                        notes.append(str(element.pitch))
                     elif type(element)==chord.Chord:
                        notes.append('.'.join(str(n) for n in element.normalOrder))
    except:
        print("Something went wrong")
        
    return notes


# In[341]:


#file_path=["schubert"]
#all_files=glob.glob("D:\AI_develpment\datasets\music_dataset/"+file_path[0]+'/*.mid',recursive=True)
notes_array=np.array([read_files(i) for i in tqdm.tqdm(flatten_files_list,position=0,leave=True )])


# In[358]:


notes_array.shape


# In[359]:


len(notes_array[27])


# In[360]:


notes=sum(notes_array,[])


# In[361]:


unique_notes=list(set(notes))
print("Unique Notes:",len(unique_notes))


# In[362]:


freq=dict(map(lambda x: (x,notes.count(x)),unique_notes))


# In[363]:


for i in range(30,100,20):
    print(i,":",len(list(filter(lambda x:x[1]>=i,freq.items()))))


# In[364]:


freq_notes=dict(filter(lambda x:x[1]>=30,freq.items()))
final_notes=[ [i for i in j if i in freq_notes] for j in notes_array]


# In[349]:


freq_notes


# In[365]:


ind2note=dict(enumerate(freq_notes))
note2ind=dict(map(reversed,ind2note.items()))


# In[366]:


print(ind2note);print(note2ind)


# In[367]:


len(final_notes)


# In[368]:


timestep=32
x=[];y=[]

for i in final_notes:   
    for j in range(0,len(i)-timestep):
        inp=i[j:j+timestep] 
        out=i[j+timestep]
        
    x.append(list(map(lambda x: note2ind[x],inp )))
    y.append(note2ind[out])
x_new=np.array(x)
y_new=np.array(y)


# In[369]:


from sklearn.model_selection import train_test_split

x_new = np.reshape(x_new,(len(x_new),timestep,1))
y_new = np.reshape(y_new,(-1,1))

x_train,x_test,y_train,y_test=train_test_split(x_new,y_new,test_size=0.2,random_state=42)


# In[370]:


model=Sequential()
model.add(LSTM(256,return_sequences=True,input_shape=(x_new.shape[1],x_new.shape[2]),name="1"))
model.add(Dropout(0.2))
model.add(LSTM(256,name="2"))
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu',name="3"))
model.add(Dense(len(note2ind),activation='softmax',name="4"))
model.summary()


# In[371]:


x_new.shape


# In[356]:


#compile the model using Adam optimizer
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
#train the model on training sets and validate on testing sets
model.fit(x_train,y_train,
          batch_size=128,
          epochs=80,
          validation_data=(x_test,y_test))


# In[276]:


model.save("D:\AI_develpment\datasets\music_dataset\model")


# In[278]:


#Inference
model1=load_model("D:\AI_develpment\datasets\music_dataset\model")


# In[303]:


index=np.random.randint(0,len(y_test)-1)
music_pattern=x_test[index]
out_pred=[]
for i in range(200):
    music_pattern= music_pattern.reshape(1,len(music_pattern),1)
    prediction=model.predict(music_pattern)
    pred_index=np.argmax(prediction)
    
    out_pred.append(ind2note[pred_index])
    music_pattern = np.append(music_pattern,pred_index)
    #update the music pattern with one timestep ahead
    music_pattern = music_pattern[1:]
    


# In[304]:


music_pattern


# In[317]:


index=np.random.randint(0,len(y_test)-1)
music_pattern=x_test[index]
out_pred=[]
for i in range(200):
    music_pattern= music_pattern.reshape(1,len(music_pattern),1)
    prediction=model.predict(music_pattern)
    pred_index=np.argmax(prediction)
    
    out_pred.append(ind2note[pred_index])
    music_pattern = np.append(music_pattern,pred_index)
    #update the music pattern with one timestep ahead
    music_pattern = music_pattern[1:]
    
output_notes=[]

for index,pattern in enumerate(out_pred):
    if ("." in pattern) or pattern.isdigit():
        notes_in_chords=pattern.split(".")
        notes=[]
        for curr_note in notes_in_chords:
            new_note=note.Note(int(curr_note))
            new_note.storedInstrument=instrument.Piano()
            notes.append(new_note)
        
        new_chord=chord.Chord(notes)
        new_chord.offset=index
        output_notes.append(new_chord)
    else:
         new_note=note.Note(pattern)
         new_note.offset=index
         new_note.storedInstrument=instrument.Piano()
         output_notes.append(new_note)

midi_stream=stream.Stream(output_notes)
midi_stream.write("midi",fp="D:\AI_develpment\Auto_music_generation\output\pred_music")


# In[310]:


"." in out_pred[10]


# In[ ]:




