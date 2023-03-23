import textgrids
import librosa
import os
from pydub import AudioSegment

# audio_path = "/Users/vickyfeng/Desktop/spice/VF19A_Cantonese_I2_20181114.wav"
# textgrid_path = "//Users/vickyfeng/Desktop/spice/VF19A_Cantonese_I2_20181114.TextGrid"

audio_ids = {"VF19A_Cantonese_I2_20181114"}

for id in audio_ids:
  audio_path = "/scratch/network/vyfeng/spice_data/" + id + ".wav"
  textgrid_path = "/scratch/network/vyfeng/spice_data/" + id + ".wav"

  audio = AudioSegment.from_wav(audio_path)
  grid = textgrids.TextGrid(textgrid_path)

  # keep track of which phones belong in which interval
  spice = []
  p_end = 0.0
  p_index = 0
  p_len = len(grid["phone"])

  dir_name = audio_path[-31:-26] 

  if not os.path.exists(dir_name):
    os.makedirs(dir_name)

  for interval in grid["utterance"]:
    if interval.text == "":
      continue
    text = ""

    # extract the audio and sample at 16Hz
    interval_audio = audio[interval.xmin*1000:interval.xmax*1000]
    interval_name = "./" + dir_name + "/" + str(interval.xmin)
    interval_audio.export(interval_name, format="wav")

    resampled = librosa.load(interval_name, sr=16000)

    # accumulate the phones that belong in this interval
    while p_end <= interval.xmax:
      phone = grid["phone"][p_index].text

      if phone == "sp":
        text += " "
      else:
        text += phone

      p_index += 1

      if p_index >= p_len:
        break
      p_end = grid["phone"][p_index].xmax
      
    utterance = {"path": audio_path, "audio": {"path": interval_name, "array": resampled, "sampling_rate": 16000}, "phones": text, "sentence": interval.text}
    spice.append(utterance)
    break
  print(resampled)