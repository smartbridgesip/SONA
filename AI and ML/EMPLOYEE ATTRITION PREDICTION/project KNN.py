
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:



import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_3032b2439ebd429882097d9e71584e90 = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='0L9JKR0MK73oxI6LktSKykXyQupRDp069_Ls4KwY16ur',
    ibm_auth_endpoint="https://iam.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.eu-geo.objectstorage.service.networklayer.com')

body = client_3032b2439ebd429882097d9e71584e90.get_object(Bucket='employeeattritionprediction-donotdelete-pr-m3c6rd3rqpexwo',Key='employee_attrition.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

dataset = pd.read_csv(body)
dataset.head()



# In[3]:


dataset.isnull().any()


# In[3]:


dataset.corr()


# In[4]:


import seaborn as sns 


# In[5]:


sns.heatmap(dataset.corr(),annot=True)


# In[4]:


x=dataset.iloc[:,0:14].values


# In[5]:


x


# In[6]:


y=dataset.iloc[:,14:].values


# In[7]:


y


# In[8]:


from sklearn.preprocessing import LabelEncoder


# In[9]:


lb=LabelEncoder()


# In[10]:


x[:,1]=lb.fit_transform(x[:,1])


# In[11]:


lb1=LabelEncoder()
x[:,3]=lb1.fit_transform(x[:,3])


# In[12]:


lb2=LabelEncoder()
x[:,8]=lb2.fit_transform(x[:,8])


# In[74]:


lb1.classes_


# In[14]:


y[:,0]=lb.fit_transform(y[:,0])


# In[15]:


y


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[18]:


x_train


# In[19]:


x_test


# In[20]:


y_train


# In[21]:


y_train=y_train.astype('int')


# In[22]:


y_test


# In[46]:


from sklearn.pipeline import Pipeline
pipeline=Pipeline([('scalar',sc),('kn',knn)])


# In[47]:


model=pipeline.fit(x_train,y_train)


# In[48]:


from sklearn.neighbors import KNeighborsClassifier


# In[49]:


knn=KNeighborsClassifier(n_neighbors=5,p=2)


# In[50]:


knn.fit(x_train,y_train)


# In[51]:


y_pred=knn.predict(x_test)


# In[52]:


y_pred


# In[53]:


y_test


# In[54]:


y_test=y_test.astype('int')


# In[55]:


from sklearn.metrics import accuracy_score


# In[56]:


print("Accuracy score:",accuracy_score(y_test,y_pred)*100,"%")


# In[57]:


from sklearn.metrics import confusion_matrix


# In[58]:


pd.DataFrame(confusion_matrix(y_test,y_pred))


# In[59]:


import sklearn.metrics as metrics
fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred)
roc_auc=metrics.auc(fpr,tpr)


# In[60]:


roc_auc


# In[61]:


plt.plot(fpr,tpr,'b',label='AUC=%0.2f' % roc_auc,color="red")
plt.legend(loc='lower right')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel("Employee data")
plt.ylabel("Employee attrition")
plt.title("Employee data vs Employee attrition")
plt.show()


# In[62]:


get_ipython().system(u'pip install watson-machine-learning-client --upgrade')


# In[63]:


from watson_machine_learning_client import WatsonMachineLearningAPIClient


# In[64]:


wml_credentials={"url":"https://eu-gb.ml.cloud.ibm.com",
                 "access_key":"814edUwidBPAo0FzUbjtNRYGqaTsRu042w19Y8JLTCoi",
                 "username":"88632294-f105-416c-8803-b3c5a1e3c1b9",
                 "password":"abb8f456-133f-4bca-91e1-01eed99c6952",
                 "instance_id":"de1c3184-06fa-4d87-9d99-2cc85c30b5b1"}


# In[65]:


client=WatsonMachineLearningAPIClient(wml_credentials)


# In[66]:


model_props={client.repository.ModelMetaNames.AUTHOR_NAME:"Durga",
             client.repository.ModelMetaNames.AUTHOR_EMAIL:"durgalakshmi2016@gmail.com",
             client.repository.ModelMetaNames.NAME:"employee attrition prediction"}
    


# In[67]:


model_s=client.repository.store_model(model,meta_props=model_props)


# In[68]:


client.repository.list()


# In[69]:


published_model_uid=client.repository.get_model_uid(model_s)


# In[70]:


published_model_uid


# In[71]:


d=client.deployments.create(published_model_uid,name="employee attrition prediction")


# In[72]:


scoring_endpoint=client.deployments.get_scoring_url(d)


# In[73]:


scoring_endpoint

