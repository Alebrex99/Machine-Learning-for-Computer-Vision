import os
import csv
import itertools
import data_set_extractor as extractor
import random

output_dir = r'C:\Users\Utente\PycharmProjects\machine_learning_DataSet\_frames'
def create_dataset(folder_path = output_dir, output_file="siamese_ds_balanced.csv"):
    dataset = []

    image_folders = os.listdir(folder_path) #'C:\Users\Utente\PycharmProjects\machine_learning_DataSet\_frames'
    ''' image_folders:
    -A_Silent_voice
    -Attack_On_Titans
    -Death_Note ...
    '''

    #VERIFICA DELLE IMMAGINI PRESE:
    images =[]
    for i in range(len(image_folders)):
        folder = image_folders[i]
        folderPath = os.path.join(folder_path, folder)
        imageFolders= os.listdir(folderPath)
        for image in imageFolders:
            images.append(image)

    #CREAZIONE DEL SIAMESE_DS
    for i in range(len(image_folders)):
        folder_a = image_folders[i]
        folder_a_path = os.path.join(folder_path, folder_a)
        if not os.path.isdir(folder_a_path):
            continue

        images_a = os.listdir(folder_a_path)
        for j in range(i, len(image_folders)):
            folder_b = image_folders[j]
            folder_b_path = os.path.join(folder_path, folder_b)
            if not os.path.isdir(folder_b_path):
                continue

            images_b = os.listdir(folder_b_path)
            for image_a in images_a:
                colab_pathA = f'/content/drive/MyDrive/Colab Notebooks/PROGETTO/_frames/{folder_a}/'
                image_a_path = os.path.join(colab_pathA, image_a)
                for image_b in images_b:
                    #print(image_a)
                    colab_pathB = f'/content/drive/MyDrive/Colab Notebooks/PROGETTO/_frames/{folder_b}/'
                    image_b_path = os.path.join(colab_pathB, image_b)
                    label = 0 if folder_a == folder_b else 1 #0 se sono lo stesso anime, 1 se sono anime diversi
                    dataset.append(((image_a_path, image_b_path), label))
    #conteggio etichette
    count_label_0 = sum(1 for _, label in dataset if label==0)
    count_label_1 = sum(1 for _, label in dataset if label == 1)
    print(count_label_0, count_label_1)

    # Creazione dei possibili accoppiamenti dell'etichetta 1 (lista)
    label_1_combinations = [((img1, img2), label) for (img1, img2), label in dataset if label == 1]
    # Calcolo del numero massimo di accoppiamenti da campionare
    max_samples = min(count_label_1, count_label_0) #ovvero sono le etichette con 0 (stesso anime)

    # Campionamento casuale degli accoppiamenti dell'etichetta 1
    selected_label_1_combinations = random.sample(label_1_combinations, max_samples)

    # Creazione dei possibili accoppiamenti dell'etichetta 0
    label_0_combinations = [((img1, img2), label) for (img1, img2), label in dataset if label == 0]
    for i in label_0_combinations[600000:600011]:
        print(i)

    # Campionamento casuale degli accoppiamenti dell'etichetta 0
    #selected_label_0_combinations = random.sample(label_0_combinations, max_samples) #non occorre, tanto max_samples = campioni con 0
    selected_label_0_combinations = label_0_combinations

    # Unione dei dataset bilanciati : cambia l'ordine delle righe rispetto al dataset originale (ordine di acquisizione)
    balanced_dataset = selected_label_1_combinations + selected_label_0_combinations

    #VERSIONE PER CERCARE DI MANTNERE L'ORDINE ORIGINALE
    '''
    # Creazione dei possibili accoppiamenti dell'etichetta 1 (lista)
    label_1_combinations = [(idx, ((img1, img2), label)) for idx, ((img1, img2), label) in enumerate(dataset) if label == 1]
    # Calcolo del numero massimo di accoppiamenti da campionare
    max_samples = min(count_label_1, count_label_0)  # ovvero sono le etichette con 0 (stesso anime)

    # Campionamento casuale degli accoppiamenti dell'etichetta 1
    selected_label_1_combinations = random.sample(label_1_combinations, max_samples)

    # Creazione dei possibili accoppiamenti dell'etichetta 0
    label_0_combinations = [(idx, ((img1, img2), label)) for idx, ((img1, img2), label) in enumerate(dataset) if
                            label == 0]
    for i in label_0_combinations[600000:600011]:
        print(i)
    # Campionamento casuale degli accoppiamenti dell'etichetta 0
    # selected_label_0_combinations = random.sample(label_0_combinations, max_samples) #non occorre, tanto max_samples = campioni con 0
    selected_label_0_combinations = label_0_combinations
    balanced_dataset = []
    # Aggiungi gli elementi con etichetta 1 nel loro ordine originale
    for idx in selected_label_1_combinations:
        balanced_dataset.append(dataset[idx[0]])

    # Aggiungi gli elementi con etichetta 0 nel loro ordine originale
    for idx in selected_label_0_combinations:
        balanced_dataset.append(dataset[idx[0]])
    '''

    count_label_balanced_0 = sum(1 for _, label in balanced_dataset if label == 0)
    count_label_balanced_1 = sum(1 for _, label in balanced_dataset if label == 1)

    print(count_label_balanced_0, count_label_balanced_1)

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ImageA', 'ImageB', 'Label'])
        for data in balanced_dataset:
            writer.writerow([data[0][0], data[0][1], data[1]])


# Esempio di utilizzo
folder_path = r'C:\Users\Utente\PycharmProjects\machine_learning_DataSet\_frames'  # Inserisci il percorso delle cartelle delle immagini
output_file = 'siamese_ds_balanced.csv'  # Specifica il nome del file di output




def main():
    create_dataset(folder_path, output_file)

    print('Done')

if __name__ == "__main__": #se lancia un eccezione esci ed interrompi flusso video
    try:
        main()
    except KeyboardInterrupt: #interruzione flusso video tramite tastiera
        exit(0)
