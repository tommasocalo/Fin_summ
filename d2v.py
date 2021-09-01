from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import json
import os
import logging
import ast

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
if __name__ == '__main__':
    total_files = [os.path.join(root, name)
             for root, dirs, files in os.walk(os.getcwd()+'/Results/Total')
             for name in files
             if name.endswith((".txt"))]
    Year_files = [os.path.join(root, name)
                 for root, dirs, files in os.walk(os.getcwd()+'/Results/Year')
                 for name in files
                 if name.endswith((".txt"))]
    Quarter_files = [os.path.join(root, name)
                 for root, dirs, files in os.walk(os.getcwd()+'/Results/Quarter')
                 for name in files
                 if name.endswith((".txt"))]
    Month_files = [os.path.join(root, name)
                 for root, dirs, files in os.walk(os.getcwd()+'/Results/Month')
                 for name in files
                 if name.endswith((".txt"))]


    def d2v(files,total = False):
        i = 0
        doc = []
        dic = {}
        res = {}
        for f in files:
            with open(f, 'r') as file:
                data = file.read()
                document = ast.literal_eval(data.split(',',1)[1])
                stock = data.split(',',1)[0]
                cat = f.split('/')[-1].split('.')[0]
                ind = i
                i+=1
                doc.append(TaggedDocument(document, [ind,stock]))
                if not total:
                    if stock in dic:
                        dic[stock][cat] = ind
                    else:
                        dic[stock] = {}
                        dic[stock][cat] = ind
                if total:
                    dic[stock] ={}
        model = Doc2Vec(doc,epochs=50,vector_size=70)
        if not total:
            for k in dic:
                for v in dic[k]:
                    if k in res:
                        res[k][v] = model[dic[k][v]].tolist()
                    else:
                        res[k]={}
                        res[k][v] = model[dic[k][v]].tolist()
        if total:
            for k in dic:
                res[k] = model[k].tolist()

        return model, dic, res

    print('Total')
    model, dic, res = d2v(Year_files,total=True)
    model.save(os.getcwd()+'/Results/Total/total_model')
    with open(os.getcwd()+'/Results/Total/dic.txt', 'w') as file:
         file.write(json.dumps(dic))
    with open(os.getcwd()+'/Results/Total/res.txt', 'w') as file:
         file.write(json.dumps(res))

    print('Years')
    model, dic, res = d2v(Year_files)
    model.save(os.getcwd()+'/Results/Year/years_model')
    with open(os.getcwd()+'/Results/Year/dic.txt', 'w') as file:
         file.write(json.dumps(dic))
    with open(os.getcwd()+'/Results/Year/res.txt', 'w') as file:
         file.write(json.dumps(res))

    print('Quarter')
    model, dic, res = d2v(Quarter_files)
    model.save(os.getcwd()+'/Results/Quarter/quarter_model')
    with open(os.getcwd()+'/Results/Quarter/dic.txt', 'w') as file:
         file.write(json.dumps(dic))
    with open(os.getcwd()+'/Results/Quarter/res.txt', 'w') as file:
         file.write(json.dumps(res))

    print('Month')
    model, dic, res = d2v(Month_files)
    model.save(os.getcwd()+'/Results/Month/Month_model')
    with open(os.getcwd()+'/Results/Month/dic.txt', 'w') as file:
         file.write(json.dumps(dic))
    with open(os.getcwd()+'/Results/Month/res.txt', 'w') as file:
         file.write(json.dumps(res))
