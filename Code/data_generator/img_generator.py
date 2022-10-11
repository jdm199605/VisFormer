import plotly
import os
import pandas as pd
import plotly.io as pio
import plotly.express as px
import json
import numpy as np
import csv
import argparse
import plotly.graph_objects as go
from func_timeout import func_set_timeout

@func_set_timeout(5)
def create_fig(xsrc, ysrc, t, path):
    if t == 'scatter':
        if xsrc and ysrc:
            fig = px.scatter(x=xsrc, y=ysrc)
        elif xsrc == None:
            fig = px.scatter(y = ysrc)
        elif ysrc == None:
            fig = px.scatter(x = xsrc)
    elif t == 'line':
        if xsrc and ysrc:
            fig = px.line(x=xsrc, y=ysrc)
        elif xsrc == None:
            fig = px.line(y = ysrc)
        elif ysrc == None:
            fig = px.line(x = xsrc)
    elif t == 'bar':
        if xsrc and ysrc:
            fig = px.bar(x=xsrc, y=ysrc)
        elif xsrc == None:
            fig = px.bar(y = ysrc)
        elif ysrc == None:
            fig = px.bar(x = xsrc)
    else:
        if xsrc and ysrc:
            fig = px.pie(values=xsrc, labels=ysrc)
        elif xsrc == None:
            fig = px.pie(labels=ysrc)
        elif ysrc == None:
            fig = px.pie(values=xsrc)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.write_image(path)

parser = argparse.ArgumentParser()
parser.add_argument('--data_size',type=int,default=70000)
parser.add_argument('--NSR',type=int,default=1)

data_size = parser.parse_args().data_size
NSR = parser.parse_args().NSR

dataset_path = '../data/dataset.tsv'
df = pd.read_table(dataset_path, sep = '\t').iloc[:data_size].reset_index(drop=True)
path = '../data/image/'
record_path = '../data/vpl_record/'
output_file_name = '../data/vpl_dataset.tsv'
headers = ['table_data','img_info']
f = open(output_file_name, 'w')
output_file = csv.writer(f, delimiter='\t')
output_file.writerow(headers)
major_type = ['scatter','line','bar','pie']
    
if not os.path.exists(path):
    os.mkdir(path)

if not os.path.exists(record_path):
    os.mkdir(record_path)

Num = 0
for i in range(len(df)):
    print ('Generating visualizations for table {}'.format(i+1))
    table_data = pd.read_table(df['table_data'][i],sep=',')
    pairs = eval(df['pairs'][i])
    types = eval(df['types'][i])
    rows = df['num_rows'][i]
    num_fields = df['num_fields'][i]
    
    table_data.replace([np.inf,-np.inf],np.nan, inplace=True)
    table_data.fillna('ffill',inplace=True)
    table_data.fillna('bfill',inplace=True)
                  
    img_info = []
    vis = pairs.copy()
    for j in range(len(vis)):
        vis[j].append(types[j])
    num = 0
                   
    for x, y, t in vis:
        if x < df['num_fields'][i]:
            xsrc = list(table_data[table_data.columns[x]])
        else:
            xsrc = None
        if y < df['num_fields'][i]:
            ysrc = list(table_data[table_data.columns[y]])
        else:
            ysrc = None
        try:
            vis_type = major_type[t]
            img_path = path + df['fid'][i] + '_' + str(num) +'.jpeg'
            create_fig(xsrc, ysrc, vis_type,img_path)
            img_info.append([img_path,1,df['table_data'][i],[x,y,t]])
            num += 1
        except:
            continue
        NEG = major_type.copy()
        NEG.remove(major_type[t])
        for n_type in NEG:
            if [x, y, major_type.index(n_type)] in vis:
                continue
            try:
                img_path = path + df['fid'][i] + '_n_' + str(num) +'.jpeg'
                create_fig(xsrc, ysrc, vis_type,img_path)
                img_info.append([img_path,0,df['table_data'][i],[x,y,major_type.index(n_type)]])
                num += 1
            except:
                continue
    
    rnd = np.random.randint(0,num_fields+1,(NSR,2))
    neg_list = []
    
    for nx, ny in rnd:
        nt = np.random.randint(0,4)
        if [nx,ny] in vis or [nx,ny,nt] in neg_list:
            continue
        if nx < df['num_fields'][i]:
            xsrc = table_data[table_data.columns[nx]]
            xsrc = list(table_data[table_data.columns[nx]])
        else:
            xsrc = None
        if ny < df['num_fields'][i]:
            ysrc = table_data[table_data.columns[ny]]
            ysrc = list(table_data[table_data.columns[ny]])
        else:
            ysrc = None
        try:
            vis_type = major_type[nt]
            img_path = path + df['fid'][i] + '_n_' + str(num) +'.jpeg'
            create_fig(xsrc, ysrc, vis_type,img)
            img_info.append([img_path,0,df['table_data'][i],[nx,ny,nt]])
            num += 1
            neg_list.append([nx,ny,nt])
        except:
            continue
   
    if len(img_info) > 0:
        record_name = record_path+'record_'+str(Num)+'.tsv'
        output_file.writerow([df['table_data'][i],record_name)
        record_file = csv.writer(open(record_name, 'w'), delimiter='\t')
        record_headers = ['img_path','label','table_data','vis']
        record_file.writerow(record_headers)
        record_file.writerows(img_info)
        Num+=1
        
        
    
        
    
    


