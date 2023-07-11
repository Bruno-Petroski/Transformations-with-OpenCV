import  cv2 as cv
import numpy as np

vid = cv.VideoCapture(0)

while True:

    # imagem normal
    ret, frame = vid.read()
    cv.imshow('Normal', frame)
    
    # Deixar a imagem com ruido
    laplacian = cv.Laplacian(frame, cv.CV_64F)
    laplacian = np.uint8(laplacian)
    cv.imshow('Ruido', laplacian)
    
    # Deixar apenas as Bordas
    Borda = cv.Canny(frame, 225, 225)
    cv.imshow('Borda', Borda)
    
    # Deixar a imagem Cinza
    Cinza = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('Cinza', Cinza)
    
    # Borrado
    Borrado = cv.GaussianBlur(frame, (3,3), cv.BORDER_DEFAULT) # Tamanho do Kernel 3, 5, 7, 9, 11.
    cv.imshow('Borrado', Borrado)
    
    # Separa BGR
    B,G,R = cv.split(frame)
    cv.imshow('Blue', B)
    cv.imshow('Red', R)
    cv.imshow('Green', G)

    # Deixa a imagem em HVS
    HVS = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    cv.imshow('HVS', HVS)

    # Rotaciona a imagem
    rows,cols = frame.shape[:2]
    M = cv.getRotationMatrix2D((cols/2,rows/2),270,1)  # 90,180,270
    dst = cv.warpAffine(frame,M,(cols,rows))
    cv.imshow('Invertido', dst)

    # Simple Threshold
    ret,thresh_binary = cv.threshold(Cinza,127,255,cv.THRESH_BINARY)
    ret,thresh_binary_inv = cv.threshold(Cinza,127,255,cv.THRESH_BINARY_INV)
    ret,thresh_trunc = cv.threshold(Cinza,127,255,cv.THRESH_TRUNC)
    ret,thresh_tozero = cv.threshold(Cinza,127,255,cv.THRESH_TOZERO)
    ret,thresh_tozero_inv = cv.threshold(Cinza,127,255,cv.THRESH_TOZERO_INV)

    Nomes = ['Binario','Binario Inv', 'Trunc', 'Tozero', 'Tozero Inv']
    images = thresh_binary,thresh_binary_inv,thresh_trunc,thresh_tozero,thresh_tozero_inv
    for i in range(5):
        cv.imshow(Nomes[i],images[i])

    # Adaptive Threshold
    thresh_mean = cv.adaptiveThreshold(Cinza, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    thresh_gaussian = cv.adaptiveThreshold(Cinza, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    Nomes =['Adaptive Threshold Mean','Adaptive Threshold Gaussian']
    images = thresh_mean,thresh_gaussian
    for i in range(2):
        cv.imshow(Nomes[i],images[i])
    
    key = cv.waitKey(50)
    if key == ord('q'):
        break
