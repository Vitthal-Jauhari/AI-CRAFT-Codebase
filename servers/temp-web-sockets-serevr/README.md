-  FFMPEG Command to convert raw to wav audio with 44100 bit rate

- ffmpeg -f s32le -ar 44100 -ac 1 -i audio.raw output.wav

- ffplay output.wav

- ffplay -f s32le -ar 44100 -af "volume=50" audio.raw  