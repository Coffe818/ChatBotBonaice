# ChatBotBonaice
version: Python 3.12.6

si tienes varias versiones de python asegurate que se te vallan a instalar en el correcto para eso puedees usar
`py -<VERSION> -m pip <LBRERIA>`, ejemplo `py -3.12 -m pip install nltk`
instalar : 
 * pip3 install torch torchvision torchaudio. Tutorial(https://www.youtube.com/watch?v=2RkK73h0BGY)
 * pip install numpy
 * pip install torch
 * pip install nltk 
 * una vez intalado nltk crea un archivo .py y escribe esto 
    ```
    import nltk
    nltk.download('punkt')
    ```
    ya despues de que se ejecute sin errores lo puedes borrar

En caso de que siga marcando error el nltk al no encontrar 'punkt' intenta usar nltk.download('punkt_tab')

