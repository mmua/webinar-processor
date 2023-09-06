# Webinar Processor

I want to create webinar processing pipeline in python.
Functions are:
* Download video from youtube
* Create video subtitles with openai whisper
* perform speaker diarization 
* speaker detection - detect known speakers
* Create streaming video player with subtitles below video syncronized with video location - when clicked on text video starts playing from linked location
* Create video content summary in text form
* detect slides in video and insert them into subtitles

## Example
"Коучинговый стиль управления как инструмент современного руководителя"

https://youtu.be/mKqDUYekM3M

## Download video from youtube

```
$ webinar_processor yt-download --help
Usage: webinar_processor yt-download [OPTIONS] URL PATH

  Downloads YouTube streams to specified directory

Options:
  --help  Show this message and exit.

```

## Транскрипция записи
### Install whisper.cpp

```
git clone https://github.com/ggerganov/whisper.cpp.git && cd whisper.cpp
mkdir build && cd build && cmake .. && make
./models/download-ggml-model.sh large
```

### Подготовка аудио-дорожки

```
ffmpeg -i video/coaching/Коучинг\ как\ инструмент\ руководителя.mp4 -ar 16000  audio/Коучинг\ как\ инструмент\ руководителя.wav
```

### Распознавание

```
time ./build/bin/main -m models/ggml-large.bin -l ru --no-timestamps -f Коучинг\ как\ инструмент\ руководителя.wav -of Коучинг\ как\ инструмент\ руководителя -otxt
```

## Транскрипция с аннотированием
### Установка пакетов python

```
pip3 install openai-whisper pywhispercpp pyannote-audio
```

# Работа с webinar_processor

## Загрузка видео
webinar_processor yt-download https://youtu.be/mKqDUYekM3M video/coaching/

## Транскрипция с диаризацией
webinar_processor transcribe /home/webinar/whisper.cpp/models/ggml-large.bin video/coaching/Коучинг\ как\ инструмент\ руководителя.mp4  video/coaching/transcript.json

# Идеи по улучшению качества
Существующие проблемы:
* ошибки в распознавании речи
* ошибки в диаризации

## Ошибки в распознавании речи
Препроцессинг аудио
Использование fine-tuning модели whisper
Прогон результатов через ChatGPT для исправления ошибок

## Ошибки в диаризации
* Подготовка embedding-ов известных нам спикеров
* Подбор моделей embedding-ов
