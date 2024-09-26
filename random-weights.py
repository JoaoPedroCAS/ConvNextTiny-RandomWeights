import os
import numpy as np
from PIL import Image
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny
import random
import time
import gc
import psutil

# Define the path to your dataset
DATASET_PATH = '/home/joaopedrocas/Projetos/feature-exctraction-random-weights/textures/'
RESULTS_PATH = '/home/joaopedrocas/Projetos/feature-exctraction-random-weights/results/'
BATCH_SIZE = 1
N_COMPONENTS = 100
# Define transformations for image preprocessing
TRASNFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ConvNeXt-Tiny model and remove the classification layer
class ConvNeXtTinyFeatureExtractor(nn.Module):
    def __init__(self):
        super(ConvNeXtTinyFeatureExtractor, self).__init__()
        self.model = convnext_tiny(weights=None)
        self.model = nn.Sequential(*list(self.model.children())[:-2])  # Remove classification layer
        #print(self.features)

    def _initialize_weights(self):
        a_std_range = (-0.015, 0.015)
        random.seed(int(time.time()))
        
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                a, b = self._get_uniform_range(a_std_range)
                nn.init.uniform_(module.weight, a=a, b=b)
                if module.bias is not None:
                    a, b = self._get_uniform_range(a_std_range)
                    nn.init.uniform_(module.bias, a=a, b=b)
        print("Inicializado")

    def _get_uniform_range(self, std_range):
        a = random.uniform(*std_range)
        b = random.uniform(*std_range)
        return min(a, b), max(a, b)

    def forward(self, x):
        return self.model(x)
    
    def remove_blocks(self):
        #print("Removendo bloco...")
        blocks = list(self.model[-1][-1].children())
        if blocks:
            blocks.pop()
            self.model[-1][-1] = nn.Sequential(*blocks)
        else:
            print("Não removeu")

    def number_of_blocks(self):
        return len(list(self.model[-1][-1])) if len(list(self.model[-1].children())) > 0 else 0
        
    def number_of_sequentials(self):
        return len(list(self.model[-1]))
    

    def remover_sequential(self):
        #print("Removendo sequential...")
        sequentials = list(self.model[-1].children())
        if sequentials:
            sequentials.pop()
            self.model[-1] = nn.Sequential(*sequentials)
    
    def printModel(self):
        print(self.model)

def load_images_from_folder(folder, transform):
    images = []
    labels = []
    for label, subfolder in enumerate(os.listdir(folder)):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = transform(img)
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    return images, labels


def extract_features_in_batches(model, images_tensor, BATCH_SIZE, device):
    #print(model)
    model.eval()
    features = []
    with torch.no_grad():
        for i in range(0, len(images_tensor), BATCH_SIZE):
            batch = images_tensor[i:i+BATCH_SIZE].to(device)
            batch_features = model(batch).cpu().numpy()
            # Flatten the features from (batch_size, channels, height, width) to (batch_size, channels * height * width)
            batch_features = batch_features.reshape(batch_features.shape[0], -1)
            features.append(batch_features)
    features = np.concatenate(features, axis=0)
    print("Extraido")
    return features

def generateReport(features, labels):
    lda = LinearDiscriminantAnalysis()
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    print("Report gerado")
    return report, y_test, y_pred

def saveMetrics(report, y_test, y_pred, labels, nome, camada):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = np.mean([report[str(i)]['f1-score'] for i in range(len(np.unique(labels))) if str(i) in report])
    recall = np.mean([report[str(i)]['recall'] for i in range(len(np.unique(labels))) if str(i) in report])
    precision = np.mean([report[str(i)]['precision'] for i in range(len(np.unique(labels))) if str(i) in report])
    with open(f'{RESULTS_PATH}{nome}_layers.txt', 'a') as f:
            f.write(f"Camadas removidas: {camada}\nAccuracy: {accuracy:.4f}\nF1-Score: {f1:.4f}\nRecall: {recall:.4f}\nPrecision: {precision:.4f}\n\n")
    print("Metricas salvas")

def check_memory():
    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total / (1024 ** 3)
    available_memory = memory_info.available / (1024 ** 3)
    used_memory = memory_info.used / (1024 ** 3)
    print(f"Memória RAM - Total: {total_memory:.2f} GB, Usada: {used_memory:.2f} GB, Disponível: {available_memory:.2f} GB")
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"Memória GPU - Alocada: {gpu_memory_allocated:.2f} GB, Reservada: {gpu_memory_reserved:.2f} GB")


def apply_pca(features, n_components):
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)
    return features_pca


# Load dataset
images, labels = load_images_from_folder(DATASET_PATH, TRASNFORM)
labels = np.array(labels)

# Convert list of tensors to a single tensor
images_tensor = torch.stack(images)

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNeXtTinyFeatureExtractor().to(device)

# Loop otimizado
nome = 104
camadas = 22

while True:
    gc.collect()
    for i in range(0, 100):

        # Extração de features em batch
        model._initialize_weights()
        features = extract_features_in_batches(model, images_tensor, BATCH_SIZE, device)
        features_pca = apply_pca(features, N_COMPONENTS)
            
        
        report, y_test, y_pred = generateReport(features_pca, labels)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Iteração {i}: Acurácia = {accuracy:.4f}")
        # Gerar novos pesos aleatórios
        saveMetrics(report, y_test, y_pred, labels, nome, camadas)
        
        del features, report, y_test, y_pred, accuracy
        torch.cuda.empty_cache()
        gc.collect()

    i = 0
    nome += 1
    camadas += 1
    try:
        if model.number_of_blocks() > 1:
            model.remove_blocks()
            print(f"Bloco removido. Blocos restantes: {model.number_of_blocks()}")
        elif model.number_of_sequentials() > 1:
            model.remover_sequential()
            print(f"Sequential removido. Sequentials restantes: {model.number_of_sequentials()}")
        else:
            print("Não há mais blocos ou sequenciais para remover. Encerrando.")
            break
    except Exception as e:
        print(f"Erro inesperado: {e}")
        break