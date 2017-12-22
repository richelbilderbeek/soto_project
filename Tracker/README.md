# README #

## Richel's guide

Do

```
sudo ./install
```

to install everything.

Do

```
./download_video
```

to download the video

Do 

```
./run
```

to run the project. If there is a warning it cannot find you movie file, modify `configuration_file.ini`.


## Old guide

This will start the tracking for each video file listed in config.ini and runs the correction at the end.
To use the program, type the following on the command line:

```
>> python main.py configuration_file.ini
```

Or just use `./run` from the command line. `run` assumes there is a video present in the same folder.

With the tester you can look at how the tracking performs over time. To use the tester, type the following on the command line:

```
>> python tester.py configuration_file.ini
```

 - `configuration_file.ini` can be used to change settings and video files.
 - Also you can have multiple config files, `config_mouse.ini`:

```
>> python main.py config_mouse.ini
```

## Errors

### `ImportError: No module named scipy`

Cause:

```
./run
```

Solution:

```
sudo pip install scipy
```

### `ImportError: No module named cv2`

Cause:

```
./run
```

Solution:

```
sudo pip install opencv-python
```

### `ImportError: No module named sklearn.cluster`

Cause:

```
./run
```

Solution:

```
sudo pip install scikit-learn
```

### `ImportError: No module named matplotlib.pyplot`

Cause:

```
./run
```

Solution:

```
sudo pip install matplotlib
```

### `ImportError: No module named _tkinter, please install the python-tk package`

Cause:

```
./run
```

Solution:

```
sudo apt-get install python-tk
```

### `AttributeError: 'module' object has no attribute 'FastFeatureDetector'`

Hypothesis: some OpenCV feature got changed, code needs to be modified for this.

### `AttributeError: 'module' object has no attribute 'cv'`

In this line:

```
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, w)
```

This means that `cv2` has no `cv` attribute, thus `cv2.cv` fails.

With some guessing, rewriting the code to:

```
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
```

works!

## Contributers

 * Pieter Bosma
 * Richel Bilderbeek

# How to get video info?


```
sudo apt-get install mediainfo
```

```
mediainfo --fullscan 3f_1.mp4
```

Results in:

```
General
Count                                    : 325
Count of stream of this kind             : 1
Kind of stream                           : General
Kind of stream                           : General
Stream identifier                        : 0
Count of video streams                   : 1
Count of audio streams                   : 1
Video_Format_List                        : MPEG-4 Visual
Video_Format_WithHint_List               : MPEG-4 Visual
Codecs Video                             : MPEG-4 Visual
Audio_Format_List                        : AAC
Audio_Format_WithHint_List               : AAC
Audio codecs                             : AAC LC
Complete name                            : 3f_1.mp4
File name                                : 3f_1
File extension                           : mp4
Format                                   : MPEG-4
Format                                   : MPEG-4
Format/Extensions usually used           : mov mp4 m4v m4a m4b m4p 3ga 3gpa 3gpp 3gp 3gpp2 3g2 k3g jpm jpx mqv ismv isma ismt f4a f4b f4v
Commercial name                          : MPEG-4
Format profile                           : Base Media
Internet media type                      : video/mp4
Codec ID                                 : isom
Codec ID                                 : isom (isom/iso2/mp41)
Codec ID/Url                             : http://www.apple.com/quicktime/download/standalone.html
CodecID_Compatible                       : isom/iso2/mp41
Codec                                    : MPEG-4
Codec                                    : MPEG-4
Codec/Extensions usually used            : mov mp4 m4v m4a m4b m4p 3ga 3gpa 3gpp 3gp 3gpp2 3g2 k3g jpm jpx mqv ismv isma ismt f4a f4b f4v
File size                                : 535078175
File size                                : 510 MiB
File size                                : 510 MiB
File size                                : 510 MiB
File size                                : 510 MiB
File size                                : 510.3 MiB
Duration                                 : 1392617
Duration                                 : 23 min 12 s
Duration                                 : 23 min 12 s 617 ms
Duration                                 : 23 min 12 s
Duration                                 : 00:23:12.617
Duration                                 : 00:23:11;14
Duration                                 : 00:23:12.617 (00:23:11;14)
Overall bit rate mode                    : VBR
Overall bit rate mode                    : Variable
Overall bit rate                         : 3073800
Overall bit rate                         : 3 074 kb/s
Frame rate                               : 29.970
Frame rate                               : 29.970 FPS
Frame count                              : 41702
Stream size                              : 1211908
Stream size                              : 1.16 MiB (0%)
Stream size                              : 1 MiB
Stream size                              : 1.2 MiB
Stream size                              : 1.16 MiB
Stream size                              : 1.156 MiB
Stream size                              : 1.16 MiB (0%)
Proportion of this stream                : 0.00226
HeaderSize                               : 1211900
DataSize                                 : 533866275
FooterSize                               : 0
IsStreamable                             : Yes
Tagged date                              : UTC 2017-09-05 14:50:37
File last modification date              : UTC 2017-10-29 10:13:51
File last modification date (local)      : 2017-10-29 11:13:51

Video
Count                                    : 338
Count of stream of this kind             : 1
Kind of stream                           : Video
Kind of stream                           : Video
Stream identifier                        : 0
StreamOrder                              : 0
ID                                       : 1
ID                                       : 1
Format                                   : MPEG-4 Visual
Commercial name                          : MPEG-4 Visual
Format profile                           : Simple@L1
Format settings, BVOP                    : No
Format settings, BVOP                    : No
Format settings, QPel                    : No
Format settings, QPel                    : No
Format settings, GMC                     : 0
Format settings, GMC                     : No warppoints
Format settings, Matrix                  : Default (H.263)
Format settings, Matrix                  : Default (H.263)
Internet media type                      : video/MP4V-ES
Codec ID                                 : 20
Codec                                    : MPEG-4V
Codec                                    : MPEG-4 Visual
Codec/Family                             : MPEG-4V
Codec/CC                                 : 20
Codec profile                            : Simple@L1
Codec settings, Packet bitstream         : No
Codec settings, BVOP                     : No
Codec settings, QPel                     : No
Codec settings, GMC                      : 0
Codec settings, GMC                      : No warppoints
Codec settings, Matrix                   : Default (H.263)
Duration                                 : 1392481
Duration                                 : 23 min 12 s
Duration                                 : 23 min 12 s 481 ms
Duration                                 : 23 min 12 s
Duration                                 : 00:23:12.481
Duration                                 : 00:23:11;14
Duration                                 : 00:23:12.481 (00:23:11;14)
Bit rate mode                            : VBR
Bit rate mode                            : Variable
Bit rate                                 : 2939125
Bit rate                                 : 2 939 kb/s
Maximum bit rate                         : 5990400
Maximum bit rate                         : 5 990 kb/s
Width                                    : 1280
Width                                    : 1 280 pixels
Height                                   : 720
Height                                   : 720 pixels
Sampled_Width                            : 1280
Sampled_Height                           : 720
Pixel aspect ratio                       : 1.000
Display aspect ratio                     : 1.778
Display aspect ratio                     : 16:9
Rotation                                 : 0.000
Frame rate mode                          : CFR
Frame rate mode                          : Constant
Frame rate                               : 29.970
Frame rate                               : 29.970 (29970/1000) FPS
FrameRate_Num                            : 29970
FrameRate_Den                            : 1000
Frame count                              : 41702
Resolution                               : 8
Resolution                               : 8 bits
Colorimetry                              : 4:2:0
Color space                              : YUV
Chroma subsampling                       : 4:2:0
Chroma subsampling                       : 4:2:0
Bit depth                                : 8
Bit depth                                : 8 bits
Scan type                                : Progressive
Scan type                                : Progressive
Interlacement                            : PPF
Interlacement                            : Progressive
Compression mode                         : Lossy
Compression mode                         : Lossy
Bits/(Pixel*Frame)                       : 0.106
Stream size                              : 511584347
Stream size                              : 488 MiB (96%)
Stream size                              : 488 MiB
Stream size                              : 488 MiB
Stream size                              : 488 MiB
Stream size                              : 487.9 MiB
Stream size                              : 488 MiB (96%)
Proportion of this stream                : 0.95609
Writing library                          : Lavc54.59.100
Writing library                          : Lavc54.59.100

Audio
Count                                    : 275
Count of stream of this kind             : 1
Kind of stream                           : Audio
Kind of stream                           : Audio
Stream identifier                        : 0
StreamOrder                              : 1
ID                                       : 2
ID                                       : 2
Format                                   : AAC
Format/Info                              : Advanced Audio Codec
Commercial name                          : AAC
Format profile                           : LC
Codec ID                                 : 40
Codec                                    : AAC LC
Codec                                    : AAC LC
Codec/Family                             : AAC
Codec/CC                                 : 40
Duration                                 : 1392617
Duration                                 : 23 min 12 s
Duration                                 : 23 min 12 s 617 ms
Duration                                 : 23 min 12 s
Duration                                 : 00:23:12.617
Duration                                 : 00:23:08:43
Duration                                 : 00:23:12.617 (00:23:08:43)
Duration_LastFrame                       : -2
Duration_LastFrame                       : -2 ms
Duration_LastFrame                       : -2 ms
Duration_LastFrame                       : -2 ms
Duration_LastFrame                       : -00:00:00.002
Bit rate mode                            : CBR
Bit rate mode                            : Constant
Bit rate                                 : 128000
Bit rate                                 : 128 kb/s
Channel(s)                               : 2
Channel(s)                               : 2 channels
Channel positions                        : Front: L R
Channel positions                        : 2/0/0
ChannelLayout                            : L R
Samples per frame                        : 1024
Sampling rate                            : 48000
Sampling rate                            : 48.0 kHz
Samples count                            : 66845616
Frame rate                               : 46.875
Frame rate                               : 46.875 FPS (1024 SPF)
Frame count                              : 65279
Compression mode                         : Lossy
Compression mode                         : Lossy
Stream size                              : 22281920
Stream size                              : 21.2 MiB (4%)
Stream size                              : 21 MiB
Stream size                              : 21 MiB
Stream size                              : 21.2 MiB
Stream size                              : 21.25 MiB
Stream size                              : 21.2 MiB (4%)
Proportion of this stream                : 0.04164
Default                                  : Yes
Default                                  : Yes
Alternate group                          : 1
Alternate group                          : 1
```


```
sudo -H pip install --upgrade pip
```