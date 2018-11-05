#ANN on Iris Dataset with adam optimizer from numpy and pandas

import numpy as np
import pandas as pd

#dataset fetching
dataset = pd.read_csv("Iris.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1:5].values

#sklearn is used only for data preprocessing
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_enc_obj = LabelEncoder()
y = label_enc_obj.fit_transform(y)
y = y.reshape(150,1)
onehot_enc_obj = OneHotEncoder(categorical_features=[0])
y = onehot_enc_obj.fit_transform(y).toarray()

#sklearn is used only for data preprocessing
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Parameters Initialization
class Layer_Utils:
    
    def __init__(self,input_shape,n_layers,layers_dims,output_shape):
        self.input_shape = input_shape
        self.n_layers = n_layers
        self.layers_dims = layers_dims
        self.output_shape = output_shape
        
    
    #Create layer_dims array
    def input_layers(self):
        
        self.layers_dims = [self.input_shape[1]]+self.layers_dims
        self.layers_dims.append(self.output_shape[1])
            
        return self.layers_dims
    
    #Parameter Initialization
    def initialization(self):
        parameters = {}
        v ={}
        s = {}
        for i in range(1,self.n_layers+2):
            parameters["W"+str(i)] = np.random.randn(self.layers_dims[i-1],self.layers_dims[i])*np.sqrt(1/self.layers_dims[i-1])
            parameters["b"+str(i)] = np.zeros((1,self.layers_dims[i]))
            
            v["dw"+str(i)] = np.zeros((self.layers_dims[i-1],self.layers_dims[i]))
            v["db"+str(i)] = np.zeros((1,self.layers_dims[i]))
            
            s["dw"+str(i)] = np.zeros((self.layers_dims[i-1],self.layers_dims[i]))
            s["db"+str(i)] = np.zeros((1,self.layers_dims[i]))
            
        return v,s,parameters
    
#ANN Model 
class ANN:
    
    def __init__(self,n_layers):
        self.n_layers = n_layers
        
    
    def relu(self,z):
        return np.maximum(0,z)
    
    def softmax(self,z):
        return np.exp(z)/np.sum(np.exp(z),axis=1)[:,np.newaxis]
    
    def relu_derivative(self,z):
        return z > 0  
    
    
    def fp_one_layer(self,X,W,b,activation=True):
        
        z = np.dot(X,W)+b
        a = self.relu(z)
        
        if activation==False:
            return z,z
        
        return a,z
    
    def forward_prop(self,X_input,parameters,n_layers):
        
        a = X_input
        activations = {}
        forward_z = {}
        activations["a0"] = a
        forward_z["z0"] = a
        
        for i in range(1,n_layers+1):
            a,z = self.fp_one_layer(a,parameters["W"+str(i)],parameters["b"+str(i)],activation=True)
            activations["a"+str(i)] = a
            forward_z["z"+str(i)] = z
        
        z,a = self.fp_one_layer(a,parameters["W"+str(n_layers+1)],parameters["b"+str(n_layers+1)],activation=False)
        y_pred = self.softmax(z)      
        
        return forward_z,activations,z,y_pred

    
    
    def loss(self,y,y_pred):
        
        m = X_train.shape[0]
        categorical_entropy_loss = (1/m)*np.sum(np.multiply(-y,np.log(y_pred)))
        
        return categorical_entropy_loss
            
    
    #Calculating dl/dz 
    def loss_der_wrt_z(self,y,y_pred):
        return y_pred-y

    
    def backprop_one_step(self,dz,z,a,w,b):
        
        m = a.shape[0]
        da = np.dot(dz,w.T)
        dw = np.dot(a.T,dz)
        db = (1/m)*np.sum(dz,axis=0)
        z_der = np.multiply(da,self.relu_derivative(z))
        
        return z_der,dw,db
    
    
    def update_parameters_adam(self,dz,parameters,forward_z,activations,t,v,s,beta1=0.9,beta2=0.99,learning_rate=0.01,epsilon=1e-8):
        
        for i in range(n_layers+1,0,-1):
            dz,dw,db = self.backprop_one_step(dz,forward_z["z"+str(i-1)],activations["a"+str(i-1)],parameters["W"+str(i)],parameters["b"+str(i)])
            
            v["dw"+str(i)] = (beta1*v["dw"+str(i)])+(1-beta1)*dw
            v["db"+str(i)] = (beta1*v["db"+str(i)])+(1-beta1)*db
            v_corrected_w = v["dw"+str(i)]/(1-np.power(beta1,t))
            v_corrected_b = v["db"+str(i)]/(1-np.power(beta1,t))
            
            s["dw"+str(i)] = (beta2*s["dw"+str(i)])+(1-beta2)*np.square(dw)
            s["db"+str(i)] = (beta2*s["db"+str(i)])+(1-beta2)*np.square(db)
            s_corrected_w = s["dw"+str(i)]/(1-np.power(beta2,t))
            s_corrected_b = s["db"+str(i)]/(1-np.power(beta2,t))
            
            parameters["W"+str(i)] = parameters["W"+str(i)]-( learning_rate* ( v_corrected_w/np.sqrt(s_corrected_w+epsilon) ) )
            parameters["b"+str(i)] = parameters["b"+str(i)]-( learning_rate* ( v_corrected_b/np.sqrt(s_corrected_b+epsilon) ) )
            
        
        return parameters
        
        
        
    def model(self,X_train,y_train,parameters,v,s,n_epochs = 1000):
        
        for i in range(1,n_epochs+1):
            forward_z,activations,z,y_pred = self.forward_prop(X_train,parameters,self.n_layers)
            total_loss = self.loss(y_train,y_pred)
            print("Loss in epoch " + str(i+1) + "=" +str(total_loss))
            dz = self.loss_der_wrt_z(y_train,y_pred)
            parameters = self.update_parameters_adam(dz,parameters,forward_z,activations,i,v,s)
            
        return parameters
    
    
    def prediction(self,parameters,X_test):
        _,_,_,a = self.forward_prop(X_test,parameters,self.n_layers)
        b = np.zeros_like(a)
        b[np.arange(len(a)), a.argmax(1)] = 1
        
        return b
    
    def check_accuracy(self,y,y_pred):
        m = y.shape[0]
        temp = np.equal(y,y_pred)
        temp = np.sum(temp,axis=1)
        sum = np.count_nonzero(temp==3)
        
        probab = sum/m
        
        return probab*100
 

#Taking Input from users regarding number of layers and layer_dims
n_layers = int(input("Enter Number of Layers:->"))
layer_dims = []
for i in range(0,n_layers):
    layer_dims.append(int(input("Enter number of neurons in " + str(i+1) + ":->")))
    
    
layer_util_obj = Layer_Utils(X_train.shape,n_layers,layer_dims,y_train.shape)
layer_dims = layer_util_obj.input_layers()
v,s,parameters = layer_util_obj.initialization()


ann = ANN(n_layers)
parameters = ann.model(X_train,y_train,parameters,v,s,4000)
y_pred = ann.prediction(parameters,X_test)
print(ann.check_accuracy(y_test,y_pred))
        

    
