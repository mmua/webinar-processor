# Webinar Processor

I want to create webinar processing pipeline in python.
Functions are:
* Download video from youtube
* Create video subtitles with openai whisper
* perform speaker diarization 
* speaker detection - detect known speakers
* for undetected speakers detect gender and generate mascot appropriately
* Create streaming video player with subtitles below video syncronized with video location - when clicked on text video starts playing from linked location
* Create video content summary in text form
* detect slides in video and insert them into subtitles

## Введение
За основу взят пост https://vas3k.club/post/18916/#2-Ustanavlivaem-bibliot

Первый опыт был позитивный, но на одном из видео whisper.cpp зациклился и сошел с ума. Как вариант, можно было бы нарезать аудио на сегменты, но стандартный whisper с задачей справился гораздо лучше. Поэтому решено было использовать его. Для ускорения - использовать aws g5 ноду.

Диаризация спикеров делается с помощью pyannote - https://github.com/yinruiqing/pyannote-whisper

Любопытно будет повторить: https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/adapting_pretrained_pipeline.ipynb

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
```
```

## Загрузка видео
webinar_processor yt-download https://youtu.be/mKqDUYekM3M video/coaching/

## Транскрипция с диаризацией
webinar_processor transcribe video/coaching/Коучинг\ как\ инструмент\ руководителя.mp4 

## Определение ключевых тем вебинара
```
webinar_processor topics video/04-gm/webinar-text.txt --output-file video/04-gm/topics.txt
```

## Отправка вебинара на сайт

## Создание текста вебинара из транскрипции
```
webinar_processor storytell --prompt-file conf/text-prompt.txt video/03-gm/transcript.json.asr --output-file video/03-gm/webinar-text.txt
```



## Создание теста по теме
```
webinar_processor summarize-with-context video/04-gm/webinar-text.txt video/04-gm/topics.txt conf/quiz-prompt.txt --output-path video/04-gm/quiz-initial.txt
```

# Идеи по улучшению качества
Существующие проблемы:
* ошибки в распознавании речи
* ошибки в диаризации

## Ошибки в распознавании речи
* Препроцессинг аудио
* Fine-tuning модели whisper
* Прогон результатов через ChatGPT для исправления ошибок

## Ошибки в диаризации
* Подготовка embedding-ов известных нам спикеров
* Подбор моделей embedding-ов

# Определение пола
Базируется на https://github.com/SuperKogito/Voice-based-gender-recognition

# Запуск на EC2

Кодирование видео на процессоре занимает x3-x6 реального времени. Для ускорения получения результата предполагается использовать EC2 instance g5.4xlarge.

## 1. Установка aws cli
https://aws.amazon.com/cli/

```
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
aws --version
```

## 2. Set Up IAM User

It's not a good practice to use the root account for AWS operations. Instead, create an IAM (Identity and Access Management) user with the necessary permissions.

1. **Login to the AWS Management Console** and navigate to the **IAM dashboard**.
2. Click on **Users** and then **Add user**.
3. Provide a username and select **Programmatic access**.
4. Click **Next** and attach necessary permissions. For simplicity and to only manage EC2, attach the **AmazonEC2FullAccess** policy. (Remember to follow the principle of least privilege in real-world applications.)
6. Complete the user creation. At the end of this process, you'll be provided an **Access Key ID** and a **Secret Access Key**. Keep these credentials secure and do not share them.

## 3. Using Temporary Session Tokens
For a secure way to provide credentials for only a session, use **AWS's Security Token Service (STS)** to generate temporary session tokens.

First, configure AWS CLI with the IAM user credentials you just created:
```
aws configure
```

Use the `sts get-session-token` command:
```
aws sts get-session-token --duration-seconds 3600
```

This command will return temporary credentials that last for an hour (3600 seconds). The output will include a temporary access key, secret access key, session token, and expiration.

Set these temporary credentials as environment variables:
```
export AWS_ACCESS_KEY_ID=YOUR_TEMP_ACCESS_KEY
export AWS_SECRET_ACCESS_KEY=YOUR_TEMP_SECRET_ACCESS_KEY
export AWS_SESSION_TOKEN=YOUR_TEMP_SESSION_TOKEN
```

Now, any AWS CLI command you run in this session will use these temporary credentials.

Here is bash script to interactive read secrets

```
#!/bin/bash

echo "Enter AWS Access Key ID:"
read -r AWS_ACCESS_KEY_ID

echo "Enter AWS Secret Access Key:"
read -rs AWS_SECRET_ACCESS_KEY

# Exporting the credentials as environment variables
export AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"

# Changing the prompt
export PS1="\[\033[36m\][AWS-TEMP-CREDS]\[\033[m\] \[\033[32m\]\w\[\033[m\] $ "

# Spawning a new shell with the custom prompt
bash --norc
```

## 4. Import ssh public key for EC2 instances
```
aws ec2 import-key-pair --key-name "my-key" --public-key-material fileb://~/.ssh/my-key.pub
```

## 5. Run EC2 Instance with custom image

You need the Image ID (AMI ID) of the image you created. You can list all your available AMIs using:

```
aws ec2 describe-images --owners self
```

Launch the Instance:
Use the run-instances command. Replace ami-xxxxxx with your AMI ID and your-key-name with your EC2 key pair name.

```
aws ec2 run-instances --image-id ami-xxxxxx --count 1 --instance-type g5.xlarge --key-name your-key-name
```

## Установка пакетов в образ

```
sudo apt install -y python3-venv ffmpeg
```


# Установка словарей spacy
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download ru_core_news_md


## Refactoring

high-ROI refactoring opportunities:

### 1. Command Naming Consistency
- Some commands use cmd_ prefix (cmd_quiz.py, cmd_topics.py) while others don't (transcribe.py, download.py)
- High ROI because it improves maintainability and makes the codebase more predictable
- Suggestion: Standardize on either using or not using the cmd_ prefix

### 2. Configuration Management
- Currently using .env for configuration but could benefit from a more structured approach
- Consider introducing a dedicated config module with:
- Default configurations
- Environment-specific overrides
- Type validation for config values
- High ROI because it reduces runtime errors and makes configuration more maintainable

### 3. Error Handling and Logging
- Looking at the command files, there's opportunity to implement consistent error handling
- Add structured logging throughout the application
- High ROI because it makes debugging and monitoring much easier

### 4. Testing Infrastructure
- I notice a `tests` directory but would need to examine its contents
- Consider adding:
    - Unit test fixtures
    - Integration tests for commands
    - Mock responses for external services (OpenAI, etc.)
- High ROI because it ensures reliability and makes future changes safer

### 5. Dependencies Isolation
- The OpenAI and other external service integrations could be better isolated
- Consider implementing a service layer pattern
- High ROI because it makes the codebase more maintainable and easier to update when APIs change

### 6. Command Pattern Standardization
- Looking at various command files, there's opportunity to standardize their structure
- Consider introducing a base command class with common functionality
- High ROI because it reduces code duplication and makes adding new commands easier

### 7. Type Hints and Documentation
- Add type hints throughout the codebase
- Improve function and class documentation
- High ROI because it improves code maintainability and helps catch errors early

### 8. Utils Module Organization
- The utils directory could be better organized by functionality
- Consider breaking it into more specific modules (e.g., io, text, media)
- High ROI because it makes the codebase more navigable