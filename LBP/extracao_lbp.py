# Importar o OpenCV para abrir imagens e transformar para escala de cinza
import cv2, os, re

# Ferramentas para calcular o histograma
from scipy.stats import itemfreq
# Para extrair o vetor de características
from skimage.feature import local_binary_pattern

# Aqui você tem que definir o diretório completo com barra pra direita
dir_origem = "C:/Users/Admin/Desktop/LBP/att_faces/"
#dir_origem = ("att_faces/")

# Função para ordenar string e int
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

numPessoas = 10

# Lista os arquivos da pasta e ordena
conteudo = os.listdir(dir_origem)
conteudo = natural_sort(conteudo)

# Vetor para treinamento do modelo e rótulos
train_feat = []
# Criação das labens (rótulos/classes) para o classificador
train_label = []

''' Será usado para o classificador '''
# Vetor para teste de predict
test_feat = []
# Criação das labens (rótulos/classes) para o classificador
test_label = []

# Para cada pasta de pessoa listada...
for pessoa in conteudo:
    
    # Esse aqui é o nome da pasta da pessoa... vamos usar como label
#    print(diretorios)
    
    # Não esquece de adicionar a barra pra direita no final do caminho
    dir_pessoas = dir_origem + pessoa + '/'
    
    '''
    # Como tem um README lá, tem que verificar se o endereço é uma pasta
    # Se não for, passa para o próximo
    '''
    if not os.path.isdir(dir_pessoas): continue
    
    # Lista o conteúdo das pessoas
    conteudo_pessoas = os.listdir(dir_pessoas)
    conteudo_pessoas = natural_sort(conteudo_pessoas)
    
#    pessoas = glob.glob(dir_pessoas+"/*")
#    pessoas = natural_sort(os.listdir(dir_pessoas))
    
    cont = 0
    
    # Agora você está manipulando as imagens...
    for i in conteudo_pessoas:
    
        img_file = dir_pessoas + i
        
        img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2GRAY)
        
    #    Apenas para mostrar a imagem
#        cv2.imshow("Resultado", img)
#        cv2.waitKey(0)

        radius = 2 # raio
        points = 8 # qtd de pontos considerados como vizinhos
        method = "nri_uniform"
    
        # Extrair o LBP
        lbp = local_binary_pattern(img, points, radius, method = method)
        
        # Ver o vetor gerado
#        print("LBP:", lbp)
    
        # Calcula o histograma normalizado
        x = itemfreq(lbp)
        total_sum = sum(x[:, 1])    

#        print("Histograma:", total_sum)
    
#        print("Soma total:", sum(total_sum))
        
        '''
        Aqui você vai fazer o seguinte:
        as 05 primeiras imgs vão para o train_feat e as outras para test_feat
        
        Faz um if else aqui que conta até 5 pra separar quando vai pra um e 
        quando vai para outro.
        '''
        if cont < 5:
        
        # Adiciona o vetor de características extraído num outro vetor para
        # ser usado no classificador
            train_feat.append(x[:, 1]/total_sum)
        
        
       
            '''
            Agora você tembém precisa adicionar a classe/rótulo/label da imagem
            Eu disse que tinha que ser número, mas da com String também...
            Então vamos usar o nome da pasta como label: s1, s2, s3, ...
        
            No teste também tem que adicionar, mas vai ser pra gente contar quantas
            o classificador acertou.
            '''
        
            train_label.append(pessoa)
            cont = cont + 1
            
        else:
            
            if cont < 10:
            
                test_feat.append(x[:, 1]/total_sum)
            
                test_label.append(pessoa)
                
                cont = cont + 1

        
'''
Depois de extrair todas as características e separar metada em cada vetor
(train_feat e test_feat), vamos criar um objeto do nosso classificador.

Neste caso estou dizendo que ele vai usar o kernel RBF (aquela imagem que te
mostrei no Skype que tinha também linear e polinomial eu acho...)

probability=True é só pra ver a % de certeza dele de cada imagem para
cada classe
'''
# Importar o classificador SVM
from sklearn import svm, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


'''
Variedade de classificadores interessantes que eu encontrei, porém apenas o 
KNN e o SVM se destacam, os outros possuem uma taxa de acurácia extremamente 
baixa. Talvez seja viável pegar apenas os 3 melhores e deixar os outros de lado.
'''
# Criar o objeto

#classificador_svm = svm.SVC(kernel='rbf', probability=True)
classificador_knn = KNeighborsClassifier(n_neighbors = 1)
#classificador_rfc = RandomForestClassifier(max_depth=2, random_state=0)
#classificador_dtc = tree.DecisionTreeClassifier()
#classificador_mlpc = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                    hidden_layer_sizes=(5, 2), random_state=1)


'''
Depois vc precisa criar o modelo de aprendizagem, que é onde o classificador
vai tentar separar todas as suas amostras de treinamento de acordo com a classe
(label/rótulo) delas. Para isso, precisa passar dois parâmetros: o vetor e a 
classe de cada. Por isso que eu criei o train_label e coloquei a identificação
da pessoa
'''
# Treina o modelo com os vetores de características e o rótulo (classe/identificador) de cada um
classificador_knn.fit(train_feat, train_label)
    
'''
Depois de criar um modelo de aprendizagem de máquima ele vai tentar classificar
as seguites amostras...
'''
# Classifica cada amostra do seguinte vetor
predict = classificador_knn.predict(test_feat)

# Ele retorna a label que acha que é para cada uma das imagens
print("Predict: ", predict)
print("Rotulos de teste: ", test_label)

#print(cont)

a = 0
acertos = 0

'''
Enquanto a for menor que o tamanho do vetor predict, os valores de certa 
posição do vetor predict e do test_label serão comparados e, se forem iguais, 
será mostrado na tela a posição do valor no vetor predict e a variável acertos 
irá ser incrementada e mostrada junto com a porcentagem de acerto total.
'''
while(a < len(predict)):
   
    
    if(predict[a] == test_label[a]):
        print("Posição do acerto do vetor: ", a) 
        acertos += 1 
    
     
    a += 1
    
print("Número de acertos: ", acertos)
print("Porcentagem de acerto: ", ((acertos/len(predict))*100))
#print(classificador_svm.predict_proba([[200,200]]))
 
'''
O resultado (essa variável "predict") é um vetor bem parecido com aquele
test_label que a gente fez... Para saber quantas ele acertou você tem que
comparar os dois e contar quantos estão iguas. Por exemplo:

Se test_label[0] == predict[0] quer dizer que o classificador acertou a classe 
do primeiro vetor classificado
'''

'''
Depois de conseguir, tente usar outros classificadores já que vc conseguiu
usar o SVM.... todos os outros são usados da mesma forma, só precisa criar
cada um deles como foi feito aqui:
    classificador_svm = svm.SVC(kernel='rbf', probability=True)
'''