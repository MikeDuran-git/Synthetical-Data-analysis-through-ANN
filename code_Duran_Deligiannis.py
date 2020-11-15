import math
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import random as rand
from scipy.special import softmax # use built-in function to avoid numerical instability
from sklearn.model_selection import train_test_split
from statistics import mean
import sklearn
from sklearn.model_selection import train_test_split
class Utility:
    @staticmethod
    def identity(Z):
        return Z,1 
    @staticmethod
    def tanh(Z):
        """
        Z : non activated outputs
        Returns (A : 2d ndarray of activated outputs, df: derivative component wise)
        """
        A = np.empty(Z.shape)
        A = 2.0/(1 + np.exp(-2.0*Z)) - 1 # A = np.tanh(Z)
        df = 1-A**2
        return A,df
    @staticmethod
    def sigmoid(Z):
        A = np.empty(Z.shape)
        A = 1.0 / (1 + np.exp(-Z))
        df = A * (1 - A)
        return A,df
    @staticmethod
    def relu(Z):
        A = np.empty(Z.shape)
        A = np.maximum(0,Z)
        df = (Z > 0).astype(int)
        return A,df
    @staticmethod
    def softmax(Z):
        return np.exp(Z)/sum(np.exp(Z))
        #return softmax(Z, axis=0) # from scipy.special
    @staticmethod
    def cross_entropy_cost(y_hat, y):
        n  = y_hat.shape[1]
        ce = -np.sum(y*np.log(y_hat+1e-9))/n
        return ce
    """
    Explication graphique du MSE:
    https://towardsdatascience.com/coding-deep-learning-for-beginners-linear-regression-part-2-cost-function-49545303d29f
    """
    @staticmethod
    def MSE_cost(y_hat, y):
        mse = np.square(np.subtract(y_hat, y)).mean()
        return mse



class NeuralNet:
    def __init__(self,X_tr=None,y_tr=None,X_te=None,y_te=None,hidden_layer_sizes=(4,),activation='identity',learning_rate=0.1,epoch=100):
        from statistics import mean
        #initialisation des variables
        self.activation_predict=activation
        self.activation=activation
        self.hidden_layer_sizes=hidden_layer_sizes
        self.learning_rate=learning_rate
        self.epoch=epoch
        self.A  = [None]*(len(self.hidden_layer_sizes)+1)
        self.df = [None]*(len(self.hidden_layer_sizes)+1)
        self.Z  = [None]*(len(self.hidden_layer_sizes)+1)
        self.erreur_train=[None]* X_tr.shape[0] # on met la qte de samble
        self.erreur_test=[None]*X_te.shape[0]
        self.err_tr=[]
        self.err_te=[]
        #initialisation des poids
        print("-----------------initialisation des poids----------------------")
        self.weights_initialization(X_tr,y_tr)
        #on fait une boucle prenant les elements individuellements
        print("-----------------lancement de l'entrainement-------------------")
        while self.epoch >0: 
            print("epoch = "+ str(self.epoch))
            #la partie ENTRAINEMENT
            for i in range(0,X_tr.shape[0]): # pour chaque sample 
                #on transpose les éléments
                x = X_tr[i:i+1].to_numpy().transpose()
                y = y_tr[i:i+1].to_numpy().transpose()
                #on optient l'erreur de la propagation vers l'avant
                self.erreur_train[i]=self.feed_forward(x,y)[0]
                #on fait la retropropagation
                self.__backward_pass(x,y)
            #on calcule la moyenne des erreurs d'ENTRAINEMENT
            erreur_train=mean(self.erreur_train)
            #LA PARTIE TEST
            for i in range(0,X_te.shape[0]):
                self.func_activation(activation)
                x = X_te[i:i+1].to_numpy().transpose()
                y = y_te[i:i+1].to_numpy().transpose()
                self.erreur_test[i]=self.feed_forward(x,y)[0]
                #c'est la phase de test donc pas de retropropagation
            #on calcule la moyene d'erreurs de TEST
            erreur_test=mean(self.erreur_test)        
            #on melanche les elements afin d'empecher l'overfitting
            X_tr, y_tr = sklearn.utils.shuffle(X_tr, y_tr)
            self.epoch-=1
            #on les mets les moyennes dans des tableaux, pour l'affichage apres
            self.err_tr.append(erreur_train)
            self.err_te .append(erreur_test)
        #on effectue l'affichage de nos données
        print("-------------------fin de l'entrainement-----------------------")
        self.epoch=epoch
        self.afficher_evolution_epoque(epoch)
        pass
    def weights_initialization(self, X_tr, y_tr):
        #print( "\n---------------------------- weights_initialization ---------------------------------" )
        #ce sont les poids on va faire  3 matrices poids et 3 vecteurs de biais
        # premier w1
        length=len(self.hidden_layer_sizes)       
        self.weight=[]
        self.bias=[]
        w1=np.random.uniform(-1,1,(self.hidden_layer_sizes[0] , X_tr.shape[1]))
        self.weight.append(w1)
        self.bias.append(np.random.uniform(-1,1,(self.hidden_layer_sizes[0] ,1)))
        #les poids de la couche cachée
        for i in range(0,length-1):
            line=self.hidden_layer_sizes[i+1]
            column=self.hidden_layer_sizes[i]
            self.weight.append(np.random.uniform(-1,1,(line,column)))
            self.bias.append(np.random.uniform(-1,1,(line,1)))
        #les poids de la couche de sortie
        line = y_tr.shape[1]
        column=self.hidden_layer_sizes[length-1]
        self.weight.append(np.random.uniform(-1,1,(line,column)))
        self.bias.append(np.random.uniform(-1,1,(line,1)))
        pass
    def feed_forward(self,x,y):
        #print( "\n---------------------------- feed_forwarding- ---------------------------------" )
        import numpy as np
        self.func_activation(self.activation_predict)
        #Couche Entrée
        Z = np.dot(self.weight[0],x) + self.bias[0]
        A=self.activation(Z)        
        #Les couches cachées
        self.A[0]=A[0]
        self.df[0]=A[1]
        for layer in range(1,len(self.hidden_layer_sizes)):
            Z = np.dot(self.weight[layer],A[0]) + self.bias[layer]
            A=self.activation(Z)
            self.Z[layer]=Z
            self.df[layer]=A[1]
            self.A[layer]=A[0]
        #Couche Sortie
        last=len(self.hidden_layer_sizes)
        Z3=np.dot(self.weight[last],A[0]) + self.bias[last]
        #la dernier Z (activation softmax)
        self.func_activation('softmax')     
        Y_pred=self.activation(Z3)
        self.Z[last]=Z3
        self.A[last]=Y_pred
        #calcul de la fonction de cout de cette instance
        error=Utility.cross_entropy_cost(Y_pred,y)
        return error,Y_pred
        pass
    def __backward_pass(self,x,y):
       # print( "\n----------------------------__backward_pass---------------------------------" )
        import numpy as np
        #on parcours chaque instance
        delta = [None] * (len(self.hidden_layer_sizes) + 1)
        dW    = [None] * (len(self.hidden_layer_sizes) + 1)
        db    = [None] * (len(self.hidden_layer_sizes) + 1)
        L=len(self.hidden_layer_sizes)
        delta[L] = self.A[L]-y
        dW[L]    = delta[L]*self.A[L-1].transpose()
        db[L]    = delta[L]
        for l in range( L-1 ,-1 ,-1):
            delta[l] = np.multiply(np.dot(self.weight[l+1].transpose() , 
            delta[l+1] ), self.df[l])
            if l == 0: # A[l-1] correspond aux _entrées_ du réseau
                dW[l]=np.dot(delta[l],x.transpose())
            else:
                dW[l] =np.dot( delta[l], self.A[l-1].transpose())
            db[l] = delta[l]
        for l in range(0,L+1):
            self.weight[l] = self.weight[l] - self.learning_rate*dW[l]
            self.bias[l] = self.bias[l] - self.learning_rate*db[l]
        pass
    def func_activation(self,activation):
        if activation=='tanh':
            self.activation=Utility.tanh
        elif activation=='sigmoid':
            self.activation=Utility.sigmoid
        elif activation=='relu':
            self.activation=Utility.relu
        elif activation=='softmax':
            self.activation=Utility.softmax  
        elif activation=='cross_entropy_cost':
            self.activation=Utility.cross_entropy_cost   
        elif activation=='MSE_cost':
            self.activation=Utility.MSE_cost
        elif activation=='identity':
            self.activation=Utility.identity
    def afficher_evolution_epoque(self,epoch):
        plt.figure(0)
        plt.plot([x for x in range(0, epoch)],self.err_tr,color="#1E90FF",label='Train')
        plt.plot([x for x in range(0, epoch)],self.err_te,color="#FF8C00",label='Test')
        plt.title("Evolution of Error during training")
        plt.legend()
        plt.xlabel("Epoch of training")
        plt.ylabel("Error")
        plt.savefig("ANN.png")
        plt.close()
        plt.show()
    def predict(self,x):
        return self.feed_forward(x,0)
        pass
def max_row(row):
    row=row.transpose()
    return np.where(row==np.amax(row))[0][0]
    pass
def calculate_model_accuracy(y_pred,y_actual):
    lenght=y_actual.shape[0]
    nbr_good_pred=0
    nbr_wrong_pred=0
    #calcul du nombre de bonne et de mauvaise predictions
    for i in range(0,lenght):
        sample_class="class-"+str(max_row(y_pred[i]))
        if sample_class==y_actual[i]:
            nbr_good_pred+=1
        else:
            nbr_wrong_pred+=1
        pass
    accuracy= (nbr_good_pred/lenght)*100
    print("accuracy = " + str(accuracy)+"%")
    fail_rate= (nbr_wrong_pred/lenght)*100
    print("fail_rate = " + str(fail_rate)+"%")
    pass
def confusion_mtx_building(confusion_mtx,y_pred,y_actual): 
    for i in range(0,X_test_final.shape[0]): # parcour de chaque sample
        predicted_class=int(max_row(y_pred[i]))# la valeur de la classe predite
        real_class=int(y_actual[i][len(y_actual[i])-1]) #la valeur de vrai classe
        confusion_mtx[real_class][predicted_class]+=1
        pass
    pass
def max_error_class_comparision(confusion_mtx):
    max_val=-1
    for i in range(0,3):
        for j in range(0,3):
            if i != j:
                if max_val<confusion_mtx[i,j]:
                    max_val=confusion_mtx[i,j]
                    params=(i,j)
            pass
    return params
    pass


if __name__ == "__main__":
    print("-------------------PREPARATION DES DONNÉES----------------------")   
    root="./"
    gaussian_df= pd.read_csv(root+"Donnees/gaussian_data.csv")
    number_of_classes_df0=gaussian_df['class'].nunique()
    # les données sont linéairement séparables car ils peuvent être séparés correctement par une frontière linéaire (hyperplan)
    #2)
    test_final_df = gaussian_df.sample(frac =0.2,random_state=42)
    gaussian_df = gaussian_df.drop(test_final_df.index)
    #b) test_final_df contiendra des données que nous utiliserons pour tester notre ANN pendant que gaussian_df contiendra des données qui entraineront l'ANN
    #3)
    X = gaussian_df.iloc[:,:-1]
    X_test_final = test_final_df.iloc[:,:-1]
    y = gaussian_df.iloc[:,-1]
    y_test_final = test_final_df.iloc[:,-1]
    #4)
    y=pd.get_dummies(y)
    #5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    print("-------------------Construction du MODEL------------------------") 
    model=NeuralNet(X_train,y_train,X_test,y_test,(4,3,2),'tanh',0.01,200)
    print("Les poids   Les bias")
    for i in range(0,len(model.weight)):
        print("Nr"+str(i)+" "+str(model.weight[i].shape) + " "+str(model.bias[i].shape))
    
    print("-------------------ANALYSE DU MODEL-----------------------")
    y_pred=[]
    for i in range(0,len(X_test_final)):
        x=X_test_final[i:i+1].to_numpy().transpose()
        y_pred.append(model.predict(x)[1].transpose())
        pass
    y_actual=y_test_final.to_numpy()
    # 3.1 RATIO
    print("-------------------------RATIO----------------------------")
    print("pour la prediction i=0 il nous renvois la classe-"+str(max_row(y_pred[0]))+" , ce qui est valide avec la valide "+ str(y_actual[0]))
    print("")
    print("calcul de l'accuracy du model")
    calculate_model_accuracy(y_pred,y_actual)
    #3.2 MATRICE DE CONFUSION
    print("-------------------MATRICE DE CONFUSION-------------------")
    print("initialisation de la matrice")
    confusion_mtx=np.zeros((number_of_classes_df0,number_of_classes_df0),dtype=int)
    print("")
    print("construction de la matrice")    
    confusion_mtx_building(confusion_mtx,y_pred,y_actual)    
    print("") 
    print("affichage de la matrice")
    print(confusion_mtx)
    class_names = ["class-0", "class-1", "class-2"]
    plt.figure(figsize = (8,8))
    plt.figure(1)
    sns.set(font_scale=2) # label size
    ax = sns.heatmap(confusion_mtx, annot=True, annot_kws={"size": 30},cbar=False, cmap='Blues', fmt='d',xticklabels=class_names, yticklabels=class_names)
    ax.set(title="", xlabel="Actual", ylabel="Predicted")
    plt.savefig("matrice de confusion.png")
    plt.show()
    print("")
    (y_pred_fail,y_actual_fail)=max_error_class_comparision(confusion_mtx)
    print("en analysant la matrice de confusion, l'erreur la plus fréquente est que la classe-"+str(y_actual_fail)+" a été predite comme étant la classe-"+str(y_pred_fail))
    print("")
    print("Cette analyse complémente bien la métrique 'pourcentage de prediction correcte', car elle prend en compte tout les cas possible, de chaque occurrence, pour toutes les classes, ainsi augmentant la validité du model. comme preuve voici le pourcentage de validité du model:")
    print("")
     #test de validité du model
    validity=(confusion_mtx[0][0]+confusion_mtx[1][1]+confusion_mtx[2][2])/len(y_actual)
    print("accuracy : "+ str(validity))
    print("")
    print("---------------Ocazou: des données pour l'analyse-------------")
    print("initialisation de y_actual et y_pred")
    y_actual = np.genfromtxt('Donnees/432-e200-b1-eta0.01-tanh_y_actual.csv', delimiter=',')
    y_pred = np.genfromtxt('Donnees/432-e200-b1-eta0.01-tanh_y_pred.csv', delimiter=',')
    print("")
    print("adaptation des données pour les fonctions")

    y_actual=y_actual.astype(int)
    my_new=[]
    for i in range(0,len(y_actual)):
        if int(y_actual[i][0])==1 and int(y_actual[i][1])==0 and int(y_actual[i][2])==0 :
            string="class-0"
        elif int(y_actual[i][0])==0 and int(y_actual[i][1])==1 and int(y_actual[i][2])==0 :
            string="class-1"
        elif int(y_actual[i][0])==0 and int(y_actual[i][1])==0 and int(y_actual[i][2])==1 :
            string="class-2"
        my_new.append(string)

    y_actual=np.array(my_new,dtype=object)
    print("-------------------------RATIO----------------------------")
    calculate_model_accuracy(y_pred,y_actual)
    print("-------------------MATRICE DE CONFUSION-------------------")
    confusion_mtx=np.zeros((number_of_classes_df0,number_of_classes_df0),dtype=int)
    confusion_mtx_building(confusion_mtx,y_pred,y_actual)    
    print("affichage de la matrice")
    print(confusion_mtx)
    class_names = ["class-0", "class-1", "class-2"]
    plt.figure(figsize = (8,8))
    plt.figure(2)
    sns.set(font_scale=2) # label size
    ax = sns.heatmap(confusion_mtx, annot=True, annot_kws={"size": 30},cbar=False, cmap='Blues', fmt='d',xticklabels=class_names, yticklabels=class_names)
    ax.set(title="", xlabel="Actual", ylabel="Predicted")
    plt.savefig("matrice de confusion2.png")
    plt.show()
    print("")
    (y_pred_fail,y_actual_fail)=max_error_class_comparision(confusion_mtx)
    print("en analysant la matrice de confusion, l'erreur la plus fréquente est que la classe-"+str(y_actual_fail)+" a été predite comme étant la classe-"+str(y_pred_fail))
    print("")
    #test de validité du model
    validity=(confusion_mtx[0][0]+confusion_mtx[1][1]+confusion_mtx[2][2])/len(y_actual)
    print("accuracy : "+ str(validity))
    print("")
