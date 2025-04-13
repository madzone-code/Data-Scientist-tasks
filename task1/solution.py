from moviepy import AudioFileClip
import os
import librosa
import noisereduce as nr
import shutil
import soundfile as sf
import speech_recognition as sr
import subprocess
import whisper


def convertation(audio_path):
    """Конвертирование звукового файла в wav."""

    # Если уже wav, то не конвертируем.
    if audio_path.lower().endswith('.wav'):
        return audio_path

    # Сохраняем файл по тому же адресу, но другим расширением.
    wav_path = os.path.splitext(audio_path)[0] + '.wav'

    audio = AudioFileClip(audio_path)
    audio.write_audiofile(wav_path, codec='pcm_s16le')
    audio.close()

    print('Файл успешно конвертирован.')
    return wav_path


def enhance_audio(audio_path):
    """Базовая функция улучшения аудио. Убирает шумы, нормализует громкость."""

    # Загрузка аудио
    audio_data, sample_rate = librosa.load(audio_path, sr=None)
    # Удаление шума
    reduced_noise = nr.reduce_noise(y=audio_data, sr=sample_rate)
    # Нормализация громкости
    normalized_audio = librosa.util.normalize(reduced_noise)
    # Сохранение обработанного аудио
    enhance_path = os.path.splitext(audio_path)[0] + '_enhance.wav'
    sf.write(enhance_path, normalized_audio, sample_rate)

    return enhance_path


def advance_enhance_audio(wav_path):
    """
    Продвинутая функция улучшения аудио.
    Убирает шумы, оставляет один голос, нормализует громкость.

    """
    # Выходной файл
    advance_enhanced_path = (
        os.path.splitext(wav_path)[0] + '_advance_enhance.wav')
    # Загрузка аудио
    audio, sample_rate = librosa.load(wav_path, sr=16000)  # 16 кГц для Whisper
    # Легкое подавление шума
    reduced_noise = nr.reduce_noise(y=audio, sr=sample_rate, stationary=False,
                                    prop_decrease=0.75)
    # Сохранение временного файла
    temp_file = "temp.wav"
    sf.write(temp_file, reduced_noise, sample_rate)
    # Выделение голоса с Demucs.
    subprocess.run(["demucs", "--two-stems=vocals", temp_file], check=True)
    vocal_file = "separated/htdemucs/temp/vocals.wav"
    # Нормализация громкости с ffmpeg
    subprocess.run([
        "ffmpeg",
        "-i", vocal_file,
        "-af", "volume=5dB",
        "-ar", "16000",
        "-y", advance_enhanced_path
    ], check=True)

    # Удаление временных файлов
    os.remove(temp_file)
    if os.path.exists("separated"):
        shutil.rmtree("separated")

    return advance_enhanced_path


def transcribe_audio(audio_path):
    """Локальное распознавание. Не требует ffmpeg."""
    recognizer = sr.Recognizer()

    print('Подождите, идет распознавание файла.')

    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language="ru-RU")
            return text
    except sr.UnknownValueError:
        return "Не удалось распознать речь"
    except sr.RequestError as e:
        return f"Ошибка сервиса: {e}"


def whisper_transcribe_audio(wav_path):
    """
    Используем нейросетевую модель от OpenAI. Доступны модели tiny, base,
    small, medium, large. Использует ffmpeg (должен быть установлен в ОС).
    """
    model = whisper.load_model("base")
    result = model.transcribe(wav_path, language="ru")
    return result["text"]


def main():
    # Задаем путь к обрабатываемому аудио файлу.
    audio_path = "task1/data/download_16.mp4"

    # Конвертируем в wav.
    wav_path = convertation(audio_path)

    # Улучшаем wav 2 способами.
    enhanced_wav_path = enhance_audio(wav_path)
    advance_enhanced_wav_path = advance_enhance_audio(wav_path)

    # Распознаем текст
    texts = {
        "original": transcribe_audio(wav_path),
        "enhanced": transcribe_audio(enhanced_wav_path),
        "advance_enhanced": transcribe_audio(advance_enhanced_wav_path),
        "whisper": whisper_transcribe_audio(wav_path),
        "whisper_enhanced": whisper_transcribe_audio(enhanced_wav_path),
        "whisper_advance_enhanced": whisper_transcribe_audio(
            advance_enhanced_wav_path),
    }

    # Записываем все тексты в файлы по тому же адресу, что и оригинал.
    base_path = os.path.splitext(audio_path)[0]
    for key, text in texts.items():
        with open(f"{base_path}_{key}.txt", "w", encoding="utf-8") as file:
            file.write(text)


if __name__ == "__main__":
    main()
