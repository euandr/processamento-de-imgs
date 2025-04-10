import cv2
import matplotlib.pyplot as plt

# Caminho da imagem
caminho_imagem = r"C:\Users\andre\Downloads\eusssds.jpg"

# Lê a imagem
imagem = cv2.imread(caminho_imagem)

# Converte para escala de cinza
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Inverte os tons
inverso = 255 - cinza

# Aplica o efeito de desfoque
desfoque = cv2.GaussianBlur(inverso, (21, 21), 0)

# Inverte o desfoque
inverso_desfoque = 255 - desfoque

# Cria o efeito de esboço
desenho = cv2.divide(cinza, inverso_desfoque, scale=256.0)

# Exibe o resultado
plt.figure(figsize=(10, 10))
plt.imshow(desenho, cmap='gray')
plt.axis('off')
plt.title("Imagem em estilo desenho a lápis")
plt.show()
