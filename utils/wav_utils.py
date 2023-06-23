import librosa
import soundfile as sf

"""
Acá bien podemos trabajar todo como mono o trabajar en stereo pero separados los canales 
y después reconstruir la señal estereo. Osea hay que hacer funcionar el código primero para mono
y la implementación del stereo sale de eso, porque sino hay que trabajar el stereo como otra matrix
lo cual implicaría tener que refactorizar bastante si se quiere hacer la adaptación.

Aclaración para mi, ahí estoy obligando a definir un fs antes, pero es mejor usar el fs que te tire
el file que ponga el usuario. Repensar y arreglar eso.
"""

def is_stereo(file_path):
    x, fs = librosa.load(file_path, mono=False)
    if len(x.shape) > 1:
        return False
    else:
        return True

def read_wav(file_path, fs, mono=False):
    """Reads an audio wav file and convert it to mono if needed.

    Args:
        file_path (str): Path to the file
        fs (int): Sample frequency.
        mono (bool): Converts to mono. Default to false.
    
    Return:
        (audio_Left, audio_Right) (np.array): Time series of the audio for left
        and rigth channel. If mono then both are the same. 
    """

    if mono:
        audio, fs = librosa.load(file_path, sr=fs, mono=True)
        return (audio, audio)

    audio, fs = librosa.load(file_path, sr=fs, mono=False)

    #Untested para estereo pero creo que esta bien
    if len(audio.shape) > 1:
        audio_L = audio[:,0]
        audio_R = audio[:,1]

        return (audio_L, audio_R)
    else:
        return (audio, audio)

def save_wav(audio, fs, save_path):
    """Save data to an audio file in wav format.

    Args:
        audio (np.array): Audio data.
        fs (int): Sample frequency.
        save_path (str): Path with the location to save the file. 
    """
    sf.write()
    sf.write(save_path, audio, fs)
    