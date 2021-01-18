import numpy as np
import pandas as pd
import cv2
import h5py

def load_and_scale_internet(path='input/f_internet_ML.csv', f):

    internet_origin = pd.read_csv(path , index_col='index')
    internet_origin = internet_origin.to_numpy()
    internet = np.log10(internet_origin + 1)
    internet_max = internet.max(axis=1)
    internet_min = internet.min(axis=1)
    internet = (internet - internet_min[:,np.newaxis]) / (internet_max - internet_min)[:,np.newaxis]
    internet = internet.T.reshape((1488,100,100))

    dataset_name = 'internet'
    if dataset_name in f: del f[dataset_name]
    h5_dataset_internent = f.create_dataset(dataset_name, internet.shape, dtype='f')
    h5_dataset_internent[:] = internet[:]
    internet = None

    dataset_name = 'internet_origin'
    if dataset_name in f: del f[dataset_name]
    h5_dataset_internent_origin = f.create_dataset(dataset_name, internet_origin.shape, dtype='f')
    h5_dataset_internent_origin[:] = internet_origin[:]
    internet_origin = None

    return h5_dataset_internent, h5_dataset_internent_origin, internet_min, internet_max

def load_and_scale_satelite(path='input/satelite.png', f):
    satelite = cv2.imread(path)
    satelite = (satelite - satelite.min()) / (satelite.max() - satelite.min())
    satelite = np.flip(satelite, axis=0)

    dataset_name = 'satelite'
    if dataset_name in f: del f[dataset_name]
    h5_dataset = f.create_dataset(dataset_name, satelite.shape, dtype='f')
    h5_dataset[:] = satelite[:]
    satelite = None

    return h5_dataset

def load_and_scale_social(path='input/social_pulse_ML.csv', f):
    social = pd.read_csv(path, index_col=0)
    social = social.to_numpy()
    social = (social - social.min(axis=1)[:,np.newaxis]) / (social.max(axis=1) - social.min(axis=1) + 1)[:,np.newaxis]
    social = social.T.reshape((1488,100,100))

    dataset_name = 'social'
    if dataset_name in f: del f[dataset_name]
    h5_dataset = f.create_dataset(dataset_name, social.shape, dtype='f')
    h5_dataset[:] = social[:]
    social = None

    return h5_dataset

def load_and_scale_weather(path='input/weather.csv', f):
    weather = pd.read_csv(path, index_col=0)
    weather = weather.to_numpy()
    weather = (weather - weather.min(axis=1)[:,np.newaxis]) / (weather.max(axis=1) - weather.min(axis=1) + 1)[:,np.newaxis]
    weather = weather.T.reshape((1488,100,100))

    dataset_name = 'weather'
    if dataset_name in f: del f[dataset_name]
    h5_dataset = f.create_dataset(dataset_name, weather.shape, dtype='f')
    h5_dataset[:] = weather[:]
    weather = None

    return h5_dataset

def create_space_invariant(f):
    hour = np.zeros((1488,24))
    weekday = np.zeros((1488,7))
    holiday = np.zeros((1488))
    for i in range(1488):
      hour[i, i % 24] = 1
      day = i // 24
      weekday[i,day % 7] = 1
      if day % 7 in [1,2]:
        holiday[i] = 1
      elif day in [0,54,56,61,62]:
        holiday[i] = 1

    dataset_name = 'hour'
    if dataset_name in f: del f[dataset_name]
    h5_dataset_h = f.create_dataset(dataset_name, hour.shape, dtype='f')
    h5_dataset_h[:] = hour[:]
    hour = None

    dataset_name = 'weekday'
    if dataset_name in f: del f[dataset_name]
    h5_dataset_w = f.create_dataset(dataset_name, weekday.shape, dtype='f')
    h5_dataset_w[:] = weekday[:]
    weekday = None

    dataset_name = 'holiday'
    if dataset_name in f: del f[dataset_name]
    h5_dataset_ho = f.create_dataset(dataset_name, holiday.shape, dtype='f')
    h5_dataset_ho[:] = holiday[:]
    holiday = None

    return h5_dataset_h, h5_dataset_w, h5_dataset_ho
